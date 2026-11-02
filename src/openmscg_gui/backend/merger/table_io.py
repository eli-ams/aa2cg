from __future__ import annotations

from pathlib import Path

import numpy as np


class TableData:
    def __init__(self, preamble: list[str], n_line_index: int | None, rows: np.ndarray):
        self.preamble = preamble
        self.n_line_index = n_line_index
        self.rows = rows

    def copy(self) -> "TableData":
        return TableData(list(self.preamble), self.n_line_index, self.rows.copy())


def table_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.table"))
    if not files:
        raise FileNotFoundError(f"No .table files found in {folder}")
    return files


def pair_table_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("Pair_*.table"))


def read_table(path: Path) -> TableData:
    preamble: list[str] = []
    rows: list[list[float]] = []
    n_line_index = None

    with path.open() as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) >= 4 and _looks_like_data_row(parts):
                rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            else:
                if parts and parts[0] == "N":
                    n_line_index = len(preamble)
                preamble.append(raw_line)

    if not rows:
        raise ValueError(f"No numeric table rows found in {path}")
    return TableData(preamble, n_line_index, np.array(rows, dtype=float))


def write_table(path: Path, table: TableData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def canonical_table_name(kind: str, labels: tuple[str, ...]) -> str:
    return f"{kind}_{'-'.join(labels)}.table"


def table_keyword(labels: tuple[str, ...]) -> str:
    return "-".join(labels)


def parse_table_name(path: Path) -> tuple[str, tuple[str, ...]]:
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Cannot parse table name: {path.name}")
    kind, raw_labels = stem.split("_", 1)
    labels = tuple(raw_labels.split("-"))
    if kind not in {"Pair", "Bond", "Angle", "Dihedral"}:
        raise ValueError(f"Unsupported table kind in {path.name}: {kind}")
    return kind, labels


def normalize_labels(kind: str, labels: tuple[str, ...]) -> tuple[str, ...]:
    if kind in {"Pair", "Bond"}:
        return tuple(sorted(labels))
    if kind in {"Angle", "Dihedral"}:
        reversed_labels = tuple(reversed(labels))
        return min(labels, reversed_labels)
    return labels


def resample_like(source: TableData, target_x: np.ndarray) -> TableData:
    rows = source.rows.copy()
    x = rows[:, 1]
    if len(x) == len(target_x) and np.allclose(x, target_x):
        return source.copy()
    if target_x[0] < x[0] or target_x[-1] > x[-1]:
        raise ValueError(
            "Cannot merge tables with incompatible coordinate ranges: "
            f"source {x[0]:.8f}-{x[-1]:.8f}, target {target_x[0]:.8f}-{target_x[-1]:.8f}"
        )
    new_rows = np.zeros((len(target_x), 4), dtype=float)
    new_rows[:, 0] = np.arange(1, len(target_x) + 1)
    new_rows[:, 1] = target_x
    new_rows[:, 2] = np.interp(target_x, x, rows[:, 2])
    new_rows[:, 3] = np.interp(target_x, x, rows[:, 3])
    return TableData(list(source.preamble), source.n_line_index, new_rows)


def recompute_force_from_energy(table: TableData) -> None:
    table.rows[:, 3] = -np.gradient(table.rows[:, 2], table.rows[:, 1])


def set_table_keyword(table: TableData, keyword: str) -> None:
    for index, line in enumerate(table.preamble):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("N"):
            continue
        table.preamble[index] = f"{keyword}\n"
        return
    insert_at = table.n_line_index if table.n_line_index is not None else len(table.preamble)
    table.preamble.insert(insert_at, f"{keyword}\n")
    if table.n_line_index is not None:
        table.n_line_index += 1


def apply_gaussian(table: TableData, amplitude: float, center: float, sigma: float) -> np.ndarray:
    distances = table.rows[:, 1]
    correction = amplitude * np.exp(-0.5 * ((distances - center) / sigma) ** 2)
    table.rows[:, 2] = table.rows[:, 2] + correction
    recompute_force_from_energy(table)
    return correction


def plot_tables(
    title: str,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    try:
        import matplotlib
    except ImportError:
        raise ImportError("matplotlib is required to write table plots. Install it with: pip install matplotlib")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, x, y in series:
        plt.plot(x, y, label=label, linewidth=1.8)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("coordinate")
    plt.ylabel("energy")
    plt.title(title)
    plt.legend(fontsize=8)
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
