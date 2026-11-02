"""Append-only trial checkpoints and portable active-study paths.

No pickle or executable state is loaded. Optuna is rebuilt from its public
trial/distribution serialization API. New studies never reuse output folders.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from .optimization_search import valid_score


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value, replace=False):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if not replace and path.exists():
        temporary.unlink()
        raise FileExistsError('Refusing to overwrite saved trial: {}'.format(path))
    os.replace(str(temporary), str(path))


def contained(root, relative):
    root = Path(root).resolve()
    candidate = root / relative
    if Path(relative).is_absolute() or root not in candidate.resolve().parents:
        raise ValueError('Saved optimization path escapes its study folder.')
    return candidate


def active_study(root):
    root = Path(root)
    pointer = root / 'active_study.json'
    return contained(root, read_json(pointer)['study']) if pointer.is_file() else None


def best_directory(root):
    root = Path(root)
    session = active_study(root)
    if session is not None:
        state = read_json(session / 'state.json')
        if state.get('best_path'):
            return contained(root, state['best_path'])
        return session / 'best' / 'not_available'
    return root / 'best'  # Existing projects remain readable without migration.


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def setup_fingerprint(source, settings):
    source = Path(source)
    ignored = {'log.lammps', 'optuna_stdout.log', 'simulation_data.out', 'cg.lammpstrj'}
    files = {}
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Study source contains a symlink; use ordinary setup files.')
        if path.is_file() and path.name not in ignored and 'table_plots' not in path.relative_to(source).parts:
            files[path.relative_to(source).as_posix()] = file_hash(path)
    payload = dict(files=files, settings=settings)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


@contextmanager
def study_lock(root):
    """OS-owned lock: automatically released on cancellation/process exit."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / '.study.lock').open('a+b')
    acquired = False
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            raise RuntimeError('Another optimizer is already using this system.')
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


class StudyHistory:
    def __init__(self, root, fingerprint, continuing=False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        session = active_study(self.root) if continuing else None
        if continuing and session is None:
            raise ValueError('This older run has no resumable study checkpoint. Start a new study; its existing outputs will remain untouched.')
        if session is None:
            studies = self.root / 'studies'
            studies.mkdir(exist_ok=True)
            self.path = studies / ('study_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
            self.path.mkdir()
            for name in ('history', 'reservations', 'runs', 'best'):
                (self.path / name).mkdir()
            self.state = dict(format_version=1, fingerprint=fingerprint, best_number=None,
                best_score=None, best_path=None, milestones=[], anchor_score=None)
            self.save()
        else:
            self.path = session
            self.state = read_json(session / 'state.json')
            if self.state['fingerprint'] != fingerprint:
                raise ValueError('Cannot continue: source setup, protocol, objectives or Gaussian limits changed. Start a new study to keep results comparable; existing outputs are untouched.')

    def activate(self):
        save_json(self.root / 'active_study.json', dict(study=self.path.relative_to(self.root).as_posix()), replace=True)

    def save(self):
        save_json(self.path / 'state.json', self.state, replace=True)

    def reserve(self, number, phase):
        save_json(self.path / 'reservations' / 'trial_{:04d}.json'.format(number),
                  dict(number=number, phase=phase))

    def numbers(self):
        return sorted({int(p.stem.split('_')[1]) for name in ('history', 'reservations', 'runs')
                       for p in (self.path / name).glob('trial_*')})

    def records(self):
        """An interrupted trial keeps its number; never rerun into its folder."""
        numbers = self.numbers()
        for number in range(numbers[-1] + 1 if numbers else 0):
            record = self.path / 'history' / 'trial_{:04d}.json'.format(number)
            if record.is_file():
                yield read_json(record)
            else:
                reservation = self.path / 'reservations' / record.name
                phase = read_json(reservation).get('phase') if reservation.is_file() else None
                yield dict(number=number, state='FAIL', value=None, params={}, distributions={},
                           user_attrs={'interrupted': True, 'phase': phase})

    def checkpoint(self, trial, metadata, score, optuna):
        record = dict(number=trial.number, state=trial.state.name,
            value=trial.value if trial.value is not None and math.isfinite(trial.value) else None,
            params=dict(trial.params), distributions={name: optuna.distributions.distribution_to_json(value)
                for name, value in trial.distributions.items()}, user_attrs=dict(trial.user_attrs),
            gaussians=metadata['pair_corrections'],
            valid_score=valid_score(score) if trial.state.name == 'COMPLETE' and valid_score(score) == trial.value else None,
            started=str(trial.datetime_start), finished=str(trial.datetime_complete))
        save_json(self.path / 'history' / 'trial_{:04d}.json'.format(trial.number), record)

    def restore(self, study, optuna):
        for record in self.records():
            state = getattr(optuna.trial.TrialState, record['state'])
            if record['value'] is None:
                state = optuna.trial.TrialState.FAIL
            frozen = optuna.trial.create_trial(state=state, value=record['value'] if state.name == 'COMPLETE' else None,
                params=record['params'], distributions={name: optuna.distributions.json_to_distribution(value)
                    for name, value in record['distributions'].items()}, user_attrs=record['user_attrs'])
            study.add_trial(frozen)

    def reconcile(self):
        candidates = [record for record in self.records() if record.get('valid_score') is not None]
        if not candidates:
            return
        best = min(candidates, key=lambda record: (record['valid_score'], record['number']))
        number = best['number']
        path = self.path / 'runs' / 'trial_{:04d}'.format(number)
        exported = self.path / 'best' / 'trial_{:04d}'.format(number)
        if exported.is_dir():
            path = exported
        elif self.state.get('best_number') == number and self.state.get('best_path'):
            path = contained(self.root, self.state['best_path'])
        if not path.is_dir():
            raise ValueError('The saved best trial data is missing; cannot safely continue this study.')
        audit = path / 'trial_record.json'
        if not audit.exists():
            save_json(audit, dict(trial=number, state='COMPLETE', value=best['value'],
                parameters=best['params'], user_attributes=best['user_attrs'],
                started=best.get('started'), finished=best.get('finished'), successful=True,
                recovered_from_checkpoint=True))
        self.state.update(best_number=number, best_score=best['valid_score'],
                          best_path=path.relative_to(self.root).as_posix())
        self.save()

    def best_settings(self):
        number = self.state['best_number']
        if number is None:
            return None
        record = read_json(self.path / 'history' / 'trial_{:04d}.json'.format(number))
        return {p['table']: {key: p[key] for key in ('amplitude', 'center', 'sigma')}
                for p in record['gaussians']}
