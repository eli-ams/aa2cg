from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import shutil
import subprocess

import numpy as np

from . import config
from .lammps_reader import read_simulation_data
from .property_scoring import score_properties
from .system_io import SystemInput
from .table_io import (
    apply_gaussian,
    pair_table_files,
    parse_table_name,
    plot_tables,
    read_table,
    set_table_keyword,
    table_keyword,
    write_table,
)


def optimize(systems: list[SystemInput], merged_tables_dir: Path) -> None:
    optuna = _import_optuna()
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.BEST_DIR.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(
        study_name=config.STUDY_NAME,
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(lambda trial: objective(trial, systems, merged_tables_dir), n_trials=config.N_TRIALS)

    best_trial_dir = _trial_dir(study.best_trial.number)
    _copy_tree_replace(best_trial_dir, config.BEST_DIR)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.6g}")
    print(f"Best setup copied to: {config.BEST_DIR}")


def objective(trial, systems: list[SystemInput], merged_tables_dir: Path) -> float:
    trial_dir = _trial_dir(trial.number)
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True)

    trial_tables = trial_dir / "global_tables"
    shutil.copytree(str(merged_tables_dir), str(trial_tables))
    corrections, regularization = _suggest_and_apply_global_corrections(trial, trial_tables, trial_dir / "table_plots")
    _write_trial_metadata(trial_dir, trial.number, corrections)

    scores = []
    for system in systems:
        system_dir = trial_dir / "systems" / system.name
        _prepare_system_trial(system, trial_tables, system_dir)
        scores.append(_run_and_score_system(system, system_dir))

    failed = [score for score in scores if score.get("failed")]
    if failed:
        score_value = 1.0e12
    else:
        score_value = float(np.average([score["score"] for score in scores], weights=[s.weight for s in systems]))
        score_value += config.REGULARIZATION_WEIGHT * regularization
    _write_score(trial_dir, score_value, scores, regularization)
    return score_value


def _suggest_and_apply_global_corrections(trial, tables_dir: Path, plot_dir: Path) -> tuple[list[dict[str, object]], float]:
    corrections = []
    regularization = 0.0
    for table_path in pair_table_files(tables_dir):
        table = read_table(table_path)
        old = table.rows[:, 2].copy()
        parameter = table_path.stem.replace("-", "_").replace(".", "_")
        amplitude = _suggest_float(trial, f"{parameter}_amplitude", config.GAUSSIAN_AMPLITUDE_MIN, config.GAUSSIAN_AMPLITUDE_MAX)
        center = _suggest_float(trial, f"{parameter}_center", config.GAUSSIAN_CENTER_MIN, config.GAUSSIAN_CENTER_MAX)
        sigma = _suggest_float(trial, f"{parameter}_sigma", config.GAUSSIAN_SIGMA_MIN, config.GAUSSIAN_SIGMA_MAX)
        correction = apply_gaussian(table, amplitude, center, sigma)
        regularization += float(np.sqrt(np.mean(correction * correction)))
        write_table(table_path, table)
        plot_tables(
            table_path.name,
            [
                ("deterministic", table.rows[:, 1], old),
                ("corrected", table.rows[:, 1], table.rows[:, 2]),
                ("correction", table.rows[:, 1], correction),
            ],
            plot_dir / f"{table_path.stem}.png",
        )
        corrections.append(
            {
                "table": table_path.name,
                "amplitude": float(amplitude),
                "center": float(center),
                "sigma": float(sigma),
            }
        )
    return corrections, regularization


def _prepare_system_trial(system: SystemInput, global_tables: Path, destination: Path) -> None:
    _copy_tree_replace(system.lammps_dir, destination)
    for local_name, canonical_name in system.table_map.items():
        source = global_tables / canonical_name
        if not source.is_file():
            raise FileNotFoundError(f"Merged table missing for {system.name}: {canonical_name}")
        _, local_labels = parse_table_name(Path(local_name))
        table = read_table(source)
        set_table_keyword(table, table_keyword(local_labels))
        write_table(destination / local_name, table)


def _run_and_score_system(system: SystemInput, system_dir: Path) -> dict[str, object]:
    log_path = system_dir / "optimizer_stdout.log"
    with log_path.open("w") as stdout:
        result = subprocess.run(
            config.LAMMPS_COMMAND,
            cwd=str(system_dir),
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        return {"system": system.name, "failed": True, "score": 1.0e12}

    score = score_properties(
        read_simulation_data(system_dir / config.SIMULATION_DATA_FILE),
        config.PROPERTIES,
    )
    score["system"] = system.name
    score["failed"] = False
    return score


def _suggest_float(trial, name: str, low: float, high: float) -> float:
    if hasattr(trial, "suggest_float"):
        return trial.suggest_float(name, low, high)
    return trial.suggest_uniform(name, low, high)


def _write_trial_metadata(trial_dir: Path, trial_number: int, corrections: list[dict[str, object]]) -> None:
    data = {
        "trial": trial_number,
        "created": datetime.now().isoformat(timespec="seconds"),
        "correction": "global_gaussian_pair_energy",
        "pair_corrections": corrections,
        "lammps_command": config.LAMMPS_COMMAND,
        "properties": config.PROPERTIES,
    }
    (trial_dir / "correction.json").write_text(json.dumps(data, indent=2) + "\n")


def _write_score(
    trial_dir: Path,
    score: float,
    system_scores: list[dict[str, object]],
    regularization: float,
) -> None:
    data = {
        "score": float(score),
        "regularization": float(regularization),
        "regularization_weight": float(config.REGULARIZATION_WEIGHT),
        "systems": system_scores,
    }
    (trial_dir / "score.json").write_text(json.dumps(data, indent=2) + "\n")


def _trial_dir(number: int) -> Path:
    return config.RUNS_DIR / f"trial_{number:04d}"


def _copy_tree_replace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(str(source), str(destination))


def _import_optuna():
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna is required for optimization. Install it with: pip install optuna")
    return optuna
