from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import shutil
import subprocess
import sys
import math
import uuid

from . import config
from ...simulation_protocol import prepare_trial_input, read_protocol, validate_scoring_window
from ...trial_storage import TrialStorage
from ...optimization_search import IncrementalSearch, normalize_search, valid_score
from ...trial_diagnostics import lammps_failure, outcome_reason
from ...optimization_history import (StudyHistory, study_lock, setup_fingerprint,
    active_study, contained, file_hash, read_json, save_json)
from .aa_cg_loss import (
    calculate_distributions,
    compare_distributions,
    pair_names_from_tables,
    write_reference,
)
from .lammps_reader import read_simulation_data
from .property_scoring import score_properties
from .table_tools import apply_gaussian_pair_table, pair_table_files, read_table


AA_CG_REFERENCE = None
AA_CG_PAIRS = None
SEARCH_CONTEXT = None
HISTORY = None
OPTUNA = None
ACTIVE_PHASE = None


def main() -> None:
    with study_lock(config.OPTIMIZATION_DIR):
        _run_study()


def _scientific_settings():
    text = prepare_trial_input((config.SOURCE_LAMMPS_DIR / 'input.data').read_text(encoding='utf-8'),
                               config.SIMULATION_PROTOCOL)
    try:
        protocol = read_protocol(text)[0]
    except ValueError:
        protocol = text
    structural = None
    if config.INCLUDE_AA_CG_IN_LOSS:
        structural = dict(weight=config.AA_CG_LOSS_WEIGHT, minimum=config.AA_CG_MIN,
            maximum=config.AA_CG_MAX, cutoff=config.AA_CG_CUTOFF, bins=config.AA_CG_BINS,
            topology=file_hash(config.AA_CG_TOPOLOGY), trajectory=file_hash(config.AA_CG_REFERENCE_TRAJECTORY))
    return dict(protocol=protocol, properties=config.PROPERTIES, bounds=_gaussian_bounds(), structural=structural)


def _run_study():
    global AA_CG_REFERENCE, AA_CG_PAIRS, SEARCH_CONTEXT, HISTORY, OPTUNA, ACTIVE_PHASE
    optuna = OPTUNA = _import_optuna()
    _require_source_lammps()
    search = normalize_search(config.SEARCH)
    fingerprint = setup_fingerprint(config.SOURCE_LAMMPS_DIR, _scientific_settings())
    legacy = config.CONTINUE_RUN and active_study(config.OPTIMIZATION_DIR) is None
    if legacy:
        _validate_legacy_best()
    HISTORY = StudyHistory(config.OPTIMIZATION_DIR, fingerprint, config.CONTINUE_RUN and not legacy)
    snapshot = HISTORY.path / 'source'
    if not config.CONTINUE_RUN or legacy:
        shutil.copytree(str(config.SOURCE_LAMMPS_DIR), str(snapshot))
    if not snapshot.is_dir() or setup_fingerprint(snapshot, _scientific_settings()) != fingerprint:
        raise ValueError('Study source snapshot is incomplete or has changed. Start a new study.')
    if legacy:
        _import_legacy_history(HISTORY, optuna)
    HISTORY.activate()
    HISTORY.reconcile()
    config.SOURCE_LAMMPS_DIR = snapshot
    config.WORK_DIR = HISTORY.path / 'runs'
    config.BEST_DIR = HISTORY.path / 'best'
    storage = TrialStorage(config.WORK_DIR, config.TRIAL_STORAGE,
                           protected_paths=(config.SOURCE_LAMMPS_DIR, config.BEST_DIR))
    storage.best_number, storage.best_score = HISTORY.state['best_number'], HISTORY.state['best_score']
    storage.milestones = set(HISTORY.state['milestones'])
    storage.anchor_score = HISTORY.state['anchor_score']
    SEARCH_CONTEXT = IncrementalSearch(_initial_gaussians(), _gaussian_bounds(),
                                      search['step_percent'], search['baseline_first'])
    if storage.best_number is not None:
        SEARCH_CONTEXT.accept(storage.best_number, HISTORY.best_settings(), storage.best_score)
    old_records = list(HISTORY.records())
    SEARCH_CONTEXT.proposals = sum(r.get('user_attrs', {}).get('phase') == 'refine' for r in old_records)

    if config.INCLUDE_AA_CG_IN_LOSS:
        AA_CG_PAIRS = pair_names_from_tables(config.SOURCE_LAMMPS_DIR)
        print("Calculating mapped-AA reference distributions...")
        AA_CG_REFERENCE = calculate_distributions(
            topology=config.AA_CG_TOPOLOGY,
            trajectory=config.AA_CG_REFERENCE_TRAJECTORY,
            pairs=AA_CG_PAIRS,
            minimum=config.AA_CG_MIN,
            maximum=config.AA_CG_MAX,
            cutoff=config.AA_CG_CUTOFF,
            bins=config.AA_CG_BINS,
        )
        reference_dir = HISTORY.path / ('aa_cg_reference_' + uuid.uuid4().hex[:8])
        write_reference(AA_CG_REFERENCE, reference_dir)

    print(f"Source: {config.SOURCE_LAMMPS_DIR}")
    print(f"Trials: {config.WORK_DIR}")
    print(f"LAMMPS: {' '.join(config.LAMMPS_COMMAND)}")
    print('Trial storage: {}'.format(storage.policy))
    print('Gaussian search: {} · baseline first: {} · total trials: {}'.format(
        search['mode'], search['baseline_first'],
        (config.ADD_EXPLORE + config.ADD_REFINE if config.CONTINUE_RUN else
         search['explore_trials'] + search['refine_trials'] if search['mode'] == 'explore_refine' else config.N_TRIALS)))

    sampler = optuna.samplers.TPESampler(seed=(config.RANDOM_SEED + 1009 * len(old_records)) % (2 ** 32))
    study = optuna.create_study(
        study_name=config.STUDY_NAME,
        direction="minimize",
        sampler=sampler,
    )
    HISTORY.restore(study, optuna)
    def completed(study, trial):
        _completed_trial(study, trial, storage)

    if config.CONTINUE_RUN:
        phases = [('explore', int(config.ADD_EXPLORE)), ('refine', int(config.ADD_REFINE))]
    elif search['mode'] == 'explore_refine':
        phases = [('explore', search['explore_trials']), ('refine', search['refine_trials'])]
    else:
        phases = [('refine' if search['mode'] == 'incremental' else 'explore', config.N_TRIALS)]
    if any(n < 0 for _, n in phases) or sum(n for _, n in phases) < 1:
        raise ValueError('Choose at least one additional trial.')
    for phase, count in phases:
        if not count:
            continue
        if phase == 'refine' and storage.best_number is None and (config.CONTINUE_RUN or search['mode'] != 'incremental'):
            print('Refine skipped: Explore did not produce a valid candidate. Add Explore trials to continue.')
            break
        ACTIVE_PHASE = phase
        print('{}: {} additional trials; starting at trial {}'.format(phase.capitalize(), count, len(study.trials)))
        study.optimize(objective, n_trials=count, callbacks=[completed])

    if storage.best_number is None:
        raise RuntimeError('No trial completed with finite scores and passing property limits. '
                           'Scores, settings and logs remain available; full files follow the storage policy.')
    best_trial_dir = _trial_dir(storage.best_number)
    if best_trial_dir.is_dir():
        storage._directory(storage.best_number)
        storage._tree(best_trial_dir)
        destination = HISTORY.path / 'best' / best_trial_dir.name
        if destination.exists():
            if valid_score(read_json(destination / 'score.json')) != storage.best_score:
                raise FileExistsError('Best checkpoint conflicts with saved scores; refusing to overwrite it.')
        elif storage.policy['mode'] == 'all':
            staging = destination.with_name('.' + destination.name + '_' + uuid.uuid4().hex + '.tmp')
            shutil.copytree(str(best_trial_dir), str(staging))
            staging.rename(destination)
        else:
            # Both paths are validated children of this study; same-volume rename.
            if (best_trial_dir.resolve().parent != config.WORK_DIR.resolve() or
                    destination.resolve().parent != (HISTORY.path / 'best').resolve()):
                raise ValueError('Unsafe best-checkpoint path.')
            best_trial_dir.rename(destination)
        HISTORY.state['best_path'] = destination.relative_to(HISTORY.root).as_posix()
        HISTORY.save()
    config.BEST_DIR = contained(HISTORY.root, HISTORY.state['best_path'])

    print("Done.")
    print(f"Best trial: {storage.best_number}")
    print(f"Best score: {storage.best_score:.6g}")
    print(f"Best setup copied to: {config.BEST_DIR}")


def _completed_trial(study, trial, storage):
    directory = _trial_dir(trial.number)
    data = json.loads((directory / 'score.json').read_text(encoding='utf-8'))
    score = valid_score(data) if trial.state.name == 'COMPLETE' else None
    if score != trial.value:
        score = None
    if SEARCH_CONTEXT is not None:
        metadata = json.loads((directory / 'correction.json').read_text(encoding='utf-8'))
        settings = {p['table']: {key: p[key] for key in ('amplitude', 'center', 'sigma')}
                    for p in metadata['pair_corrections']}
        if SEARCH_CONTEXT.accept(trial.number, settings, score):
            print('Incremental reference: trial {} (score {:.6g})'.format(trial.number, score))
    if HISTORY is not None:
        metadata = read_json(directory / 'correction.json')
        HISTORY.checkpoint(trial, metadata, data, OPTUNA)
    print('Trial {}: {} · {}'.format(trial.number,
          'valid' if score is not None else 'rejected', outcome_reason(data)))
    # Update the reference BEFORE storage can remove a worse/failed trial.
    storage(study, trial)
    if HISTORY is not None:
        HISTORY.state.update(best_number=storage.best_number, best_score=storage.best_score,
            milestones=sorted(storage.milestones), anchor_score=storage.anchor_score)
        if storage.best_number == trial.number:
            HISTORY.state['best_path'] = directory.relative_to(HISTORY.root).as_posix()
        HISTORY.save()


def objective(trial) -> float:
    trial_dir = _trial_dir(trial.number)
    if HISTORY is not None:
        HISTORY.reserve(trial.number, ACTIVE_PHASE)
    # Never delete or replace an existing trial directory, including interrupted ones.
    shutil.copytree(str(config.SOURCE_LAMMPS_DIR), str(trial_dir))
    input_path = trial_dir / 'input.data'
    if config.SIMULATION_PROTOCOL:
        input_path.write_text(prepare_trial_input(input_path.read_text(encoding='utf-8'),
                                                 config.SIMULATION_PROTOCOL), encoding='utf-8')

    baseline = normalize_search(config.SEARCH)['baseline_first'] and trial.number == 0
    if hasattr(trial, 'set_user_attr'):
        trial.set_user_attr('baseline', baseline)
        trial.set_user_attr('search_mode', normalize_search(config.SEARCH)['mode'])
        trial.set_user_attr('phase', ACTIVE_PHASE or ('refine' if normalize_search(config.SEARCH)['mode'] == 'incremental' else 'explore'))
    print('Trial {} · {}'.format(trial.number, 'unmodified baseline' if baseline else 'Gaussian candidate'))
    if baseline:
        # Do NOT call the corrector: even amplitude=0 used to recompute raw forces.
        corrections = [dict(table=name, **values) for name, values in _initial_gaussians().items()]
    else:
        corrections = _suggest_and_apply_pair_corrections(trial, trial_dir)

    _write_trial_metadata(trial_dir, trial.number, corrections, getattr(trial, 'user_attrs', {}))
    return _run_and_score(trial_dir)


def _suggest_and_apply_pair_corrections(trial, trial_dir: Path) -> list[dict[str, float | str]]:
    corrections = []
    plot_dir = trial_dir / "table_plots"
    proposal = None
    if ACTIVE_PHASE == 'refine' or (ACTIVE_PHASE is None and normalize_search(config.SEARCH)['mode'] == 'incremental'):
        if SEARCH_CONTEXT is None:
            raise RuntimeError('Incremental search was not initialized.')
        proposal, details = SEARCH_CONTEXT.propose(trial)
        for key, value in details.items():
            trial.set_user_attr(key, value)
    for table in pair_table_files(trial_dir):
        parameter = table.stem.replace("-", "_").replace(".", "_")
        if proposal is not None:
            values = proposal[table.name]
            if values['amplitude'] == 0:
                corrections.append(dict(table=table.name, **values))
                continue  # Preserve inactive interactions byte-for-byte.
            corrections.append(apply_gaussian_pair_table(table, plot_dir=plot_dir, **values))
            continue
        amplitude = _suggest_float(
            trial,
            f"{parameter}_amplitude",
            config.GAUSSIAN_AMPLITUDE_MIN,
            config.GAUSSIAN_AMPLITUDE_MAX,
        )
        center = _suggest_float(
            trial,
            f"{parameter}_center",
            config.GAUSSIAN_CENTER_MIN,
            config.GAUSSIAN_CENTER_MAX,
        )
        sigma = _suggest_float(
            trial,
            f"{parameter}_sigma",
            config.GAUSSIAN_SIGMA_MIN,
            config.GAUSSIAN_SIGMA_MAX,
        )
        corrections.append(
            apply_gaussian_pair_table(
                table,
                amplitude=amplitude,
                center=center,
                sigma=sigma,
                plot_dir=plot_dir,
            )
        )
    return corrections


def _gaussian_bounds():
    return dict(amplitude=(config.GAUSSIAN_AMPLITUDE_MIN, config.GAUSSIAN_AMPLITUDE_MAX),
                center=(config.GAUSSIAN_CENTER_MIN, config.GAUSSIAN_CENTER_MAX),
                sigma=(config.GAUSSIAN_SIGMA_MIN, config.GAUSSIAN_SIGMA_MAX))


def _validate_legacy_best():
    """Older JSON records can be imported only after checking their actual setup."""
    best = Path(config.OPTIMIZATION_DIR) / 'best'
    required = ('score.json', 'correction.json', 'structure.data', 'input.data')
    if not all((best / name).is_file() for name in required):
        raise ValueError('No resumable history or complete legacy best result was found. Start a new study; existing files will be preserved.')
    metadata = read_json(best / 'correction.json')
    if valid_score(read_json(best / 'score.json')) is None:
        raise ValueError('The legacy best result is invalid; start a new Explore study instead.')
    if config.INCLUDE_AA_CG_IN_LOSS or metadata.get('include_aa_cg_in_loss'):
        raise ValueError('Legacy structural-loss runs lack a reference fingerprint. Start a new study; newer studies support continuation.')
    if (best / 'bead_signature.json').is_file() and file_hash(best / 'bead_signature.json') != file_hash(config.SOURCE_LAMMPS_DIR / 'bead_signature.json'):
        raise ValueError('Bead definitions changed since the legacy run; start a new study.')
    old_protocol = read_protocol((best / 'input.data').read_text(encoding='utf-8'))[0]
    if (metadata['properties'] != config.PROPERTIES or old_protocol != _scientific_settings()['protocol'] or
            file_hash(best / 'structure.data') != file_hash(config.SOURCE_LAMMPS_DIR / 'structure.data')):
        raise ValueError('Legacy setup, masses, protocol or objectives differ from the current settings. Start a new study instead of mixing incompatible results.')
    corrections = {p['table']: p for p in metadata['pair_corrections']}
    for path in config.SOURCE_LAMMPS_DIR.glob('*.table'):
        previous = best / path.name
        if not previous.is_file():
            raise ValueError('A legacy best table is missing; cannot verify continuation.')
        if not path.name.startswith('Pair_'):
            if file_hash(path) != file_hash(previous):
                raise ValueError('Bonded tables changed since the legacy run; start a new study.')
            continue
        values = corrections[path.name]
        old_rows, result_rows = read_table(path).rows, read_table(previous).rows
        if len(old_rows) != len(result_rows):
            raise ValueError('Pair-table grids changed since the legacy run.')
        for old, result in zip(old_rows, result_rows):
            energy = old[2] + values['amplitude'] * math.exp(-.5 * ((old[1] - values['center']) / values['sigma']) ** 2)
            if not math.isclose(old[1], result[1], abs_tol=1.e-7) or not math.isclose(energy, result[2], abs_tol=1.e-6, rel_tol=1.e-7):
                raise ValueError('Source pair potentials changed since the legacy run; start a new study.')


def _import_legacy_history(history, optuna):
    root = Path(config.OPTIMIZATION_DIR)
    best = root / 'best'
    best_meta = read_json(best / 'correction.json')
    best_record = read_json(best / 'trial_record.json') if (best / 'trial_record.json').is_file() else {}
    selected = None
    for record in (root / 'trial_records').glob('run_*/trial_*/trial_record.json'):
        data = read_json(record)
        if data.get('trial') == best_meta['trial'] and data.get('started') == best_record.get('started'):
            selected = record.parent.parent
            break
    paths = list((root / 'runs').glob('trial_*'))
    if selected is not None:
        archived = list(selected.glob('trial_*'))
        last = max([int(p.name.split('_')[1]) for p in archived] + [int(best_meta['trial'])])
        paths = [p for p in paths if int(p.name.split('_')[1]) <= last] + archived
    paths.append(best)
    records = {}
    distribution = getattr(optuna.distributions, 'FloatDistribution', None) or optuna.distributions.UniformDistribution
    for path in paths:
        if not (path / 'correction.json').is_file() or not (path / 'score.json').is_file():
            continue
        metadata, score = read_json(path / 'correction.json'), read_json(path / 'score.json')
        record = read_json(path / 'trial_record.json') if (path / 'trial_record.json').is_file() else {}
        if metadata['properties'] != best_meta['properties']:
            raise ValueError('Legacy history mixes different scoring objectives; start a new study.')
        if metadata.get('effective_simulation_protocol') is not None:
            from ...simulation_protocol import normalize_protocol
            if normalize_protocol(metadata['effective_simulation_protocol']) != _scientific_settings()['protocol']:
                raise ValueError('Legacy history mixes different simulation protocols; start a new study.')
        if (path / 'input.data').is_file() and read_protocol((path / 'input.data').read_text(encoding='utf-8'))[0] != _scientific_settings()['protocol']:
            raise ValueError('Legacy trial input differs from the current protocol; start a new study.')
        if (path / 'structure.data').is_file() and file_hash(path / 'structure.data') != file_hash(config.SOURCE_LAMMPS_DIR / 'structure.data'):
            raise ValueError('Legacy history contains different structures or masses; start a new study.')
        number = int(metadata['trial'])
        params = record.get('parameters', {})
        distributions = {}
        for name, value in params.items():
            field = next((key for key in _gaussian_bounds() if name.endswith('_' + key)), None)
            limits = (-1., 1.) if name.endswith('_step') else _gaussian_bounds().get(field)
            if limits is None or not limits[0] <= value <= limits[1]:
                raise ValueError('Legacy parameter ranges are incompatible with the current search.')
            distributions[name] = optuna.distributions.distribution_to_json(distribution(*limits))
        attrs = dict(record.get('user_attributes') or metadata.get('search_trial') or {})
        attrs.setdefault('phase', 'refine' if attrs.get('search_mode') == 'incremental' else 'explore')
        value = score.get('score')
        value = float(value) if value is not None and math.isfinite(float(value)) else None
        records[number] = dict(number=number, state='COMPLETE' if value is not None else 'FAIL',
            value=value, params=params, distributions=distributions, user_attrs=attrs,
            valid_score=valid_score(score), gaussians=metadata['pair_corrections'],
            started=record.get('started', metadata.get('created')), finished=record.get('finished'))
    for number in range(max(records) + 1):
        record = records.get(number, dict(number=number, state='FAIL', value=None,
            params={}, distributions={}, user_attrs={'legacy_missing': True}))
        save_json(history.path / 'history' / 'trial_{:04d}.json'.format(number), record)
    history.state.update(best_number=int(best_meta['trial']), best_path='best',
                         best_score=valid_score(read_json(best / 'score.json')))
    history.save()


def _initial_gaussians():
    bounds = _gaussian_bounds()
    initial = {}
    for table in pair_table_files(config.SOURCE_LAMMPS_DIR):
        rows = read_table(table).rows
        candidates = rows[(rows[:, 1] >= bounds['center'][0]) & (rows[:, 1] <= bounds['center'][1])]
        center = (float(candidates[candidates[:, 2].argmin(), 1]) if len(candidates)
                  else sum(bounds['center']) / 2.)
        initial[table.name] = dict(amplitude=0., center=center, sigma=sum(bounds['sigma']) / 2.)
    return initial


def _suggest_float(trial, name: str, low: float, high: float) -> float:
    if hasattr(trial, "suggest_float"):
        return trial.suggest_float(name, low, high)
    return trial.suggest_uniform(name, low, high)


def _run_and_score(trial_dir: Path) -> float:
    log_path = trial_dir / "optuna_stdout.log"
    with log_path.open("w") as stdout:
        result = subprocess.run(
            config.LAMMPS_COMMAND,
            cwd=str(trial_dir),
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        _write_score(trial_dir, lammps_failure(log_path, trial_dir / 'structure.data', result.returncode))
        return 1.0e12

    try:
        score_data = _score_trial(trial_dir)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        _write_score(trial_dir, dict(failed=True, score=1.e12, failure_category='scoring_error',
                                     failure_reason='Could not score production output: {}'.format(exc)))
        return 1.e12
    if config.INCLUDE_AA_CG_IN_LOSS:
        aa_cg_loss, pair_losses = _score_aa_cg(trial_dir)
        score_data["property_score"] = score_data["score"]
        score_data["aa_cg_loss"] = aa_cg_loss
        score_data["aa_cg_loss_weight"] = config.AA_CG_LOSS_WEIGHT
        score_data["aa_cg_pair_losses"] = pair_losses
        score_data["score"] += config.AA_CG_LOSS_WEIGHT * aa_cg_loss
    _write_score(trial_dir, score_data)
    return score_data["score"]


def _score_aa_cg(trial_dir: Path) -> tuple[float, dict[str, float]]:
    if AA_CG_REFERENCE is None or AA_CG_PAIRS is None:
        raise RuntimeError("AA-CG reference distributions were not initialized.")
    trajectory = trial_dir / config.AA_CG_TRIAL_TRAJECTORY
    candidate = calculate_distributions(
        topology=config.AA_CG_TOPOLOGY,
        trajectory=trajectory,
        pairs=AA_CG_PAIRS,
        minimum=config.AA_CG_MIN,
        maximum=config.AA_CG_MAX,
        cutoff=config.AA_CG_CUTOFF,
        bins=config.AA_CG_BINS,
    )
    return compare_distributions(
        AA_CG_REFERENCE,
        candidate,
        trial_dir / "aa_cg",
    )


def _score_trial(trial_dir: Path) -> dict[str, object]:
    data_path = trial_dir / config.SIMULATION_DATA_FILE
    return score_properties(read_simulation_data(data_path), config.PROPERTIES)


def _write_trial_metadata(
    trial_dir: Path,
    trial_number: int,
    corrections: list[dict[str, float | str]],
    search_details=None,
) -> None:
    metadata = {
        "trial": trial_number,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_lammps_dir": str(config.SOURCE_LAMMPS_DIR),
        "lammps_command": config.LAMMPS_COMMAND,
        "correction": "gaussian_pair_energy",
        "pair_corrections": corrections,
        "include_aa_cg_in_loss": config.INCLUDE_AA_CG_IN_LOSS,
        "properties": config.PROPERTIES,
        "simulation_protocol_overrides": config.SIMULATION_PROTOCOL,
        "trial_storage": config.TRIAL_STORAGE,
        "gaussian_search": normalize_search(config.SEARCH),
        "search_trial": dict(search_details or {}),
    }
    try:
        metadata['effective_simulation_protocol'] = read_protocol(
            (trial_dir / 'input.data').read_text(encoding='utf-8'))[0]
    except ValueError:
        metadata['effective_simulation_protocol'] = None  # Custom inherited input is authoritative.
    (trial_dir / "correction.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _write_score(trial_dir: Path, score_data: dict[str, object]) -> None:
    (trial_dir / "score.json").write_text(json.dumps(score_data, indent=2) + "\n")


def _trial_dir(number: int) -> Path:
    return config.WORK_DIR / f"trial_{number:04d}"


def _require_source_lammps() -> None:
    source = config.SOURCE_LAMMPS_DIR
    if not source.is_dir():
        raise FileNotFoundError(f"Missing source LAMMPS directory: {source}")
    for name in ["input.data", "structure.data"]:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing required LAMMPS file: {source / name}")
    if not (source / "bead_signature.json").is_file():
        raise FileNotFoundError(
            f"Missing bead signature file: {source / 'bead_signature.json'}. "
            "Rerun GUI stages 1 through 4 with the current project."
        )
    pair_table_files(source)
    if config.SIMULATION_PROTOCOL:
        prepared = prepare_trial_input((source / 'input.data').read_text(encoding='utf-8'),
                                       config.SIMULATION_PROTOCOL)
        validate_scoring_window(read_protocol(prepared)[0], config.PROPERTIES)


def _import_optuna():
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna is required for this stage. Install it with: pip install optuna")
    return optuna


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
