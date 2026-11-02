from __future__ import annotations

import math
from pathlib import Path

from .structure_detection import read_lammps_structure


def inspect_trajectory(
    trajectory_path: str | Path, structure_path: str | Path | None = None
) -> dict:
    """Scan a LAMMPS dump and return display-ready statistics and explicit checks."""
    path = Path(trajectory_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory does not exist: {path}")

    timesteps: list[int] = []
    atom_counts: list[int] = []
    first_columns: list[str] = []
    first_ids: set[int] | None = None
    ids_consistent = True
    duplicate_ids = False
    finite_coordinates = True
    finite_forces = True
    force_samples = 0
    force_squared_sum = 0.0
    maximum_force = 0.0
    box_lengths: list[tuple[float, float, float]] = []
    first_molecule_count: int | None = None

    with path.open(encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            while line and not line.startswith("ITEM: TIMESTEP"):
                line = handle.readline()
            if not line:
                break
            timestep_line = handle.readline().strip()
            try:
                timesteps.append(int(float(timestep_line)))
            except ValueError as exc:
                raise ValueError(f"Invalid trajectory timestep: {timestep_line!r}") from exc
            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Malformed trajectory: NUMBER OF ATOMS header is missing.")
            atom_count = int(handle.readline().strip())
            atom_counts.append(atom_count)
            bounds_header = handle.readline().strip()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Malformed trajectory: BOX BOUNDS header is missing.")
            lengths = []
            for _ in range(3):
                values = handle.readline().split()
                if len(values) < 2:
                    raise ValueError("Malformed trajectory box bounds.")
                lengths.append(float(values[1]) - float(values[0]))
            box_lengths.append(tuple(lengths))
            atoms_header = handle.readline().strip()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError("Malformed trajectory: ATOMS header is missing.")
            columns = atoms_header.split()[2:]
            if not first_columns:
                first_columns = columns
            elif columns != first_columns:
                ids_consistent = False
            column_index = {name: index for index, name in enumerate(columns)}
            frame_ids = []
            frame_molecules = set()
            coordinate_columns = _coordinate_columns(column_index)
            has_forces = all(name in column_index for name in ("fx", "fy", "fz"))
            for _ in range(atom_count):
                values = handle.readline().split()
                if len(values) != len(columns):
                    raise ValueError("Malformed trajectory atom row.")
                if "id" in column_index:
                    frame_ids.append(int(values[column_index["id"]]))
                if "mol" in column_index:
                    frame_molecules.add(int(values[column_index["mol"]]))
                for column in coordinate_columns:
                    finite_coordinates &= math.isfinite(float(values[column_index[column]]))
                if has_forces:
                    components = [float(values[column_index[name]]) for name in ("fx", "fy", "fz")]
                    finite_forces &= all(math.isfinite(value) for value in components)
                    magnitude_squared = sum(value * value for value in components)
                    force_squared_sum += magnitude_squared
                    force_samples += 1
                    maximum_force = max(maximum_force, math.sqrt(magnitude_squared))
            if frame_ids:
                frame_set = set(frame_ids)
                duplicate_ids |= len(frame_ids) != len(frame_set)
                if first_ids is None:
                    first_ids = frame_set
                else:
                    ids_consistent &= frame_set == first_ids
            if first_molecule_count is None and frame_molecules:
                first_molecule_count = len({value for value in frame_molecules if value > 0})

    if not timesteps:
        raise ValueError(f"No LAMMPS frames were found in {path}.")

    timestep_deltas = [second - first for first, second in zip(timesteps, timesteps[1:])]
    uniform_stride = len(set(timestep_deltas)) <= 1
    monotonic = all(delta > 0 for delta in timestep_deltas)
    atom_count_consistent = len(set(atom_counts)) == 1
    box_jump = _maximum_relative_box_jump(box_lengths)
    columns = set(first_columns)
    has_positions = bool(_coordinate_columns({name: index for index, name in enumerate(first_columns)}))
    has_forces = all(name in columns for name in ("fx", "fy", "fz"))
    has_velocities = all(name in columns for name in ("vx", "vy", "vz"))
    has_molecule_ids = "mol" in columns
    checks = [
        _check("Frames", "pass" if len(timesteps) > 1 else "warning", f"{len(timesteps)} frame(s)"),
        _check(
            "Atom count",
            "pass" if atom_count_consistent else "fail",
            f"{atom_counts[0]} atoms" if atom_count_consistent else "Atom count changes between frames",
        ),
        _check(
            "Atom IDs",
            "pass" if first_ids is not None and ids_consistent and not duplicate_ids else "fail",
            "Unique and consistent between frames"
            if first_ids is not None and ids_consistent and not duplicate_ids
            else "Missing, duplicated, or inconsistent IDs",
        ),
        _check(
            "Timestep sequence",
            "pass" if monotonic and uniform_stride else "warning" if monotonic else "fail",
            "Strictly increasing with a uniform stride"
            if monotonic and uniform_stride
            else "Increasing but sampled irregularly" if monotonic else "Not strictly increasing",
        ),
        _check(
            "Coordinates",
            "pass" if has_positions and finite_coordinates else "fail",
            "Present and finite" if has_positions and finite_coordinates else "Missing or non-finite values",
        ),
        _check(
            "Forces",
            "pass" if has_forces and finite_forces else "fail",
            "Present and finite"
            if has_forces and finite_forces
                else "Not present" if not has_forces else "Contains non-finite values",
        ),
        _check(
            "Molecule IDs",
            "pass" if has_molecule_ids else "warning",
            "Present" if has_molecule_ids else "Absent; structure topology will define molecules",
        ),
        _check(
            "Box continuity",
            "pass" if box_jump <= 0.10 else "warning",
            f"Maximum frame-to-frame length change: {100.0 * box_jump:.3g}%",
        ),
    ]

    topology = {}
    if structure_path and Path(structure_path).is_file():
        structure = read_lammps_structure(structure_path)
        topology = {"atoms": len(structure.atoms), "bonds": len(structure.bonds)}
        structure_ids = set(structure.atoms)
        matches = (
            atom_count_consistent
            and atom_counts[0] == len(structure.atoms)
            and first_ids == structure_ids
        )
        checks.append(
            _check(
                "Topology alignment",
                "pass" if matches else "fail",
                "Trajectory atom IDs match the structure"
                if matches
                else "Trajectory and structure atom IDs/counts differ",
            )
        )

    return {
        "format": "LAMMPS dump",
        "file_size_bytes": path.stat().st_size,
        "frames": len(timesteps),
        "first_timestep": timesteps[0],
        "last_timestep": timesteps[-1],
        "timestep_span": timesteps[-1] - timesteps[0],
        "frame_stride": timestep_deltas[0] if timestep_deltas and uniform_stride else None,
        "atoms": atom_counts[0],
        "molecules_from_trajectory": first_molecule_count,
        "columns": first_columns,
        "box_first": list(box_lengths[0]),
        "box_last": list(box_lengths[-1]),
        "has_forces": has_forces,
        "has_velocities": has_velocities,
        "has_molecule_ids": has_molecule_ids,
        "force_rms": math.sqrt(force_squared_sum / force_samples) if force_samples else None,
        "force_max": maximum_force if force_samples else None,
        "topology": topology,
        "checks": checks,
    }


def _coordinate_columns(column_index: dict[str, int]) -> list[str]:
    result = []
    for axis in "xyz":
        for candidate in (f"{axis}u", axis, f"{axis}s"):
            if candidate in column_index:
                result.append(candidate)
                break
    return result if len(result) == 3 else []


def _maximum_relative_box_jump(box_lengths) -> float:
    maximum = 0.0
    for previous, current in zip(box_lengths, box_lengths[1:]):
        for old, new in zip(previous, current):
            if old > 0.0:
                maximum = max(maximum, abs(new - old) / old)
    return maximum


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}
