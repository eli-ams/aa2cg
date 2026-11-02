from __future__ import annotations

from pathlib import Path

import numpy as np


class TableData:
    def __init__(
        self,
        preamble: list[str],
        n_line_index: int | None,
        rows: np.ndarray,
    ):
        self.preamble = preamble
        self.n_line_index = n_line_index
        self.rows = rows


def pair_table_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("Pair_*.table"))
    if not files:
        raise FileNotFoundError(f"No Pair_*.table files found in {folder}")
    return files


def read_table(path: Path) -> TableData:
    preamble: list[str] = []
    data_rows: list[list[float]] = []
    n_line_index = None

    with path.open() as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) >= 4 and _looks_like_data_row(parts):
                data_rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            else:
                if parts and parts[0] == "N":
                    n_line_index = len(preamble)
                preamble.append(raw_line)

    if not data_rows:
        raise ValueError(f"No table data rows found in {path}")
    return TableData(preamble=preamble, n_line_index=n_line_index, rows=np.array(data_rows))


def write_table(path: Path, table: TableData) -> None:
    rows = table.rows
    preamble = list(table.preamble)
    if table.n_line_index is not None:
        preamble[table.n_line_index] = (
            f"N {len(rows)} # {rows[-1, 1]:.8f} {np.max(rows[:, 2]):.8f} {np.max(rows[:, 3]):.8f}\n"
        )

    with path.open("w") as handle:
        handle.writelines(preamble)
        if preamble and preamble[-1].strip():
            handle.write("\n")
        for index, row in enumerate(rows, start=1):
            handle.write(f"{index} {row[1]:.8f} {row[2]:.8f} {row[3]:.8f}\n")


def apply_gaussian_pair_table(
    path: Path,
    amplitude: float,
    center: float,
    sigma: float,
    plot_dir: Path,
) -> dict[str, float | str]:
    table = read_table(path)
    distances = table.rows[:, 1]
    old_energies = table.rows[:, 2].copy()
    correction = amplitude * np.exp(-0.5 * ((distances - center) / sigma) ** 2)
    energies = old_energies + correction
    forces = -np.gradient(energies, distances)

    table.rows[:, 0] = np.arange(1, len(table.rows) + 1)
    table.rows[:, 2] = energies
    table.rows[:, 3] = forces
    write_table(path, table)
    plot_path = plot_dir / f"{path.stem}.old_vs_new.png"
    plot_old_vs_new(path.name, distances, old_energies, energies, correction, plot_path)
    return {
        "table": path.name,
        "amplitude": float(amplitude),
        "center": float(center),
        "sigma": float(sigma),
        "plot": str(plot_path),
    }


def plot_old_vs_new(
    table_name: str,
    distances: np.ndarray,
    old_energies: np.ndarray,
    new_energies: np.ndarray,
    correction: np.ndarray,
    output_path: Path,
) -> None:
    try:
        import matplotlib
    except ImportError:
        raise ImportError("matplotlib is required to write old-vs-new table PNG plots.")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(distances, old_energies, label="old U(r)", linewidth=2)
    plt.plot(distances, new_energies, label="new U(r)", linewidth=2)
    plt.plot(distances, correction, label="Gaussian correction", linestyle="--")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("r")
    plt.ylabel("Energy")
    plt.title(table_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _looks_like_data_row(parts: list[str]) -> bool:
    try:
        int(float(parts[0]))
        float(parts[1])
        float(parts[2])
        float(parts[3])
    except ValueError:
        return False
    return True
