from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from . import config
from .system_io import SystemInput
from .table_io import (
    TableData,
    plot_tables,
    read_table,
    recompute_force_from_energy,
    resample_like,
    set_table_keyword,
    table_files,
    write_table,
)


class MergeResult:
    def __init__(self, tables_dir: Path, table_sources: dict[str, list[dict[str, object]]]):
        self.tables_dir = tables_dir
        self.table_sources = table_sources


def deterministic_merge(systems: list[SystemInput]) -> MergeResult:
    output = config.OUTPUT_DIR / "deterministic"
    tables_dir = output / "tables"
    plots_dir = output / "plots"
    reports_dir = output / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    inventory = _collect_inventory(systems)
    table_sources: dict[str, list[dict[str, object]]] = {}
    report_rows = []

    for canonical_name, entries in sorted(inventory.items()):
        merged, diagnostics = _merge_one_table(canonical_name, entries, tables_dir, plots_dir)
        write_table(tables_dir / canonical_name, merged)
        table_sources[canonical_name] = [
            {"system": entry["system"].name, "local_table": entry["path"].name}
            for entry in entries
        ]
        report_rows.append(diagnostics)

    _write_coverage_report(reports_dir / "coverage.csv", report_rows)
    _write_manifest(output / "manifest.json", systems, table_sources)
    return MergeResult(tables_dir=tables_dir, table_sources=table_sources)


def _collect_inventory(systems: list[SystemInput]) -> dict[str, list[dict[str, object]]]:
    inventory: dict[str, list[dict[str, object]]] = {}
    for system in systems:
        for path in table_files(system.lammps_dir):
            canonical = system.table_map[path.name]
            inventory.setdefault(canonical, []).append({"system": system, "path": path})
    return inventory


def _merge_one_table(
    canonical_name: str,
    entries: list[dict[str, object]],
    tables_dir: Path,
    plots_dir: Path,
) -> tuple[TableData, dict[str, object]]:
    reference = read_table(entries[0]["path"])
    x = reference.rows[:, 1]
    weighted_energies = np.zeros_like(x)
    weighted_forces = np.zeros_like(x)
    total_weight = 0.0
    source_series = []
    source_names = []

    for entry in entries:
        system = entry["system"]
        table = resample_like(read_table(entry["path"]), x)
        weight = float(system.weight)
        weighted_energies += weight * table.rows[:, 2]
        weighted_forces += weight * table.rows[:, 3]
        total_weight += weight
        source_series.append((system.name, table.rows[:, 1], table.rows[:, 2]))
        source_names.append(system.name)

    merged = reference.copy()
    merged.rows[:, 0] = np.arange(1, len(x) + 1)
    merged.rows[:, 2] = weighted_energies / total_weight
    merged.rows[:, 3] = weighted_forces / total_weight
    if config.MERGE_FORCE_FROM_ENERGY:
        recompute_force_from_energy(merged)
    set_table_keyword(merged, Path(canonical_name).stem.split("_", 1)[1])

    diagnostics = _diagnose(canonical_name, source_names, source_series, merged)
    plot_tables(
        canonical_name,
        source_series + [("merged", merged.rows[:, 1], merged.rows[:, 2])],
        plots_dir / f"{Path(canonical_name).stem}.png",
    )
    return merged, diagnostics


def _diagnose(
    canonical_name: str,
    source_names: list[str],
    source_series: list[tuple[str, np.ndarray, np.ndarray]],
    merged: TableData,
) -> dict[str, object]:
    rms_values = []
    max_values = []
    for _, x, energy in source_series:
        merged_energy = np.interp(x, merged.rows[:, 1], merged.rows[:, 2])
        diff = energy - merged_energy
        rms_values.append(float(np.sqrt(np.mean(diff * diff))))
        max_values.append(float(np.max(np.abs(diff))))
    rms_energy = max(rms_values) if rms_values else 0.0
    max_energy = max(max_values) if max_values else 0.0
    status = "single_source" if len(source_series) == 1 else "merged"
    if rms_energy > config.CONFLICT_RMS_ENERGY:
        status = "conflict"
    return {
        "table": canonical_name,
        "status": status,
        "sources": ";".join(source_names),
        "source_count": len(source_names),
        "max_rms_energy_to_merged": rms_energy,
        "max_abs_energy_to_merged": max_energy,
        "merged_min_energy": float(np.min(merged.rows[:, 2])),
        "merged_min_coordinate": float(merged.rows[int(np.argmin(merged.rows[:, 2])), 1]),
    }


def _write_coverage_report(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "table",
        "status",
        "sources",
        "source_count",
        "max_rms_energy_to_merged",
        "max_abs_energy_to_merged",
        "merged_min_energy",
        "merged_min_coordinate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    path: Path,
    systems: list[SystemInput],
    table_sources: dict[str, list[dict[str, object]]],
) -> None:
    data = {
        "systems": [
            {
                "name": system.name,
                "root": str(system.root),
                "lammps_dir": str(system.lammps_dir),
                "weight": system.weight,
                "label_map": system.label_map,
            }
            for system in systems
        ],
        "tables": table_sources,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
