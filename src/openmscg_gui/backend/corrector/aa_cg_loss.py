from __future__ import annotations

from pathlib import Path
import csv

import numpy as np


def pair_names_from_tables(folder: Path) -> list[tuple[str, str]]:
    pairs = []
    for path in sorted(folder.glob("Pair_*.table")):
        names = path.stem[len("Pair_") :].split("-")
        if len(names) != 2:
            raise ValueError(f"Could not read pair names from table: {path.name}")
        pair = (names[0], names[1])
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise FileNotFoundError(f"No Pair_*.table files found in {folder}")
    return pairs


def calculate_distributions(
    topology: Path,
    trajectory: Path,
    pairs: list[tuple[str, str]],
    minimum: float,
    maximum: float,
    cutoff: float,
    bins: int,
) -> dict[str, np.ndarray]:
    from mscg.cli import cgib

    if not topology.is_file():
        raise FileNotFoundError(f"AA-CG topology not found: {topology}")
    if not trajectory.is_file():
        raise FileNotFoundError(f"AA-CG trajectory not found: {trajectory}")

    distributions = {}
    for first, second in pairs:
        name = f"Pair-{first}-{second}"
        spec = f"{first},{second},min={minimum},max={maximum},bins={bins}"
        result = cgib.main(
            top=str(topology),
            traj=str(trajectory),
            cut=float(cutoff),
            pair=[spec],
            save="return",
        )[0]
        distributions[name] = np.asarray(result["data"], dtype=float)
    return distributions


def compare_distributions(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[float, dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    losses = {}
    for name, reference_data in reference.items():
        if name not in candidate:
            raise KeyError(f"Candidate distribution missing: {name}")
        candidate_data = candidate[name]
        x_reference = reference_data[:, 0]
        y_reference = reference_data[:, 1]
        y_candidate = np.interp(x_reference, candidate_data[:, 0], candidate_data[:, 1])
        denominator = float(np.mean(y_reference ** 2)) + 1.0e-12
        loss = float(np.mean((y_candidate - y_reference) ** 2) / denominator)
        losses[name] = loss
        _write_comparison_csv(output_dir / f"{name}.csv", x_reference, y_reference, y_candidate)
        _write_comparison_plot(output_dir / f"{name}.png", name, x_reference, y_reference, y_candidate)
    return float(np.mean(list(losses.values()))), losses


def write_reference(distributions: dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in distributions.items():
        with (output_dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "aa_reference"])
            writer.writerows((float(row[0]), float(row[1])) for row in data)


def _write_comparison_csv(
    path: Path,
    x: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "aa_reference", "cg_trial"])
        writer.writerows(zip(x, reference, candidate))


def _write_comparison_plot(
    path: Path,
    title: str,
    x: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    try:
        import matplotlib
    except ImportError:
        raise ImportError("matplotlib is required to write AA-CG comparison plots.")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, reference, label="mapped AA reference", linewidth=2)
    ax.plot(x, candidate, label="CG trial", linewidth=2, linestyle="--")
    ax.set_xlabel("r")
    ax.set_ylabel("distribution")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
