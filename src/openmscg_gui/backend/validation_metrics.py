from __future__ import annotations

import json
import math
from pathlib import Path
import re
from ..optimization_history import best_directory
from ..validation_sampling import normalize_sampling, sampled_indices


MAX_RDF_ATOMS = 650
PROPERTY_COLUMNS = [
    {"label": "Temperature", "column": "c_thermo_temp"},
    {"label": "Pressure", "column": "c_thermo_press"},
    {"label": "Mass density", "column": "v_sysdensity"},
    {"label": "Total energy", "column": "v_etotal"},
]
PROPERTY_ALIASES = {
    "c_thermo_temp": ("c_thermo_temp", "temp", "temperature"),
    "c_thermo_press": ("c_thermo_press", "press", "pressure"),
    "v_sysdensity": ("v_sysdensity", "sysdensity", "density", "massdensity", "rho"),
    "v_etotal": ("v_etotal", "etotal", "totalenergy", "toteng"),
}


def validate_system(
    system_root: str | Path,
    merged_candidate: str | Path | None = None,
    aa_properties_path: str | Path | None = None,
    cg_candidate: str | Path | None = None,
    sampling_settings: dict | None = None,
) -> dict:
    """Compare the mapped-AA reference with the best available CG trajectory.

    This is intentionally a compact diagnostic, not a replacement for a full
    validation campaign.  It reports an intermolecular all-bead RDF, bonded
    geometry distributions, topology counts, and a few trajectory/property
    statistics using evenly spaced frames across each selected window.
    """
    import numpy as np

    sampling = normalize_sampling(sampling_settings)
    root = Path(system_root)
    aa_path = root / "output" / "work" / "types_cg.lammpstrj"
    candidate_paths = [
        Path(merged_candidate) if merged_candidate else None,
        best_directory(root / 'optimization') / 'cg.lammpstrj',
        root / "output" / "lammps" / "cg.lammpstrj",
    ]
    # The GUI supplies its stage-aware selection explicitly so an invalidated
    # optimization cannot silently supersede a current baseline simulation.
    if cg_candidate is not None:
        candidate_paths = [Path(cg_candidate)]
    cg_path = next((path for path in candidate_paths if path is not None and path.is_file()), None)
    if not aa_path.is_file():
        raise FileNotFoundError(f"Mapped-AA reference trajectory not found: {aa_path}")
    if cg_path is None:
        raise FileNotFoundError(
            "No CG trajectory was found. Run Refine or simulate the generated LAMMPS setup "
            "externally to produce output/lammps/cg.lammpstrj. A smoke test is not a simulation."
        )

    structure_paths = [
        root / "output" / "lammps" / "structure.data",
        root / "output" / "work" / "structure.data",
    ]
    structure_path = next((path for path in structure_paths if path.is_file()), None)
    if structure_path is None:
        raise FileNotFoundError("Generated CG structure.data was not found.")

    topology = _read_topology(structure_path)
    aa_frames, aa_sampling = _read_sampled_frames(aa_path, sampling)
    cg_frames, cg_sampling = _read_sampled_frames(cg_path, sampling)
    _check_compatible(aa_frames, cg_frames, topology)

    rdf = _rdf_comparison(aa_frames, cg_frames, topology["molecules"], np)
    bonds = _geometry_comparison(
        aa_frames, cg_frames, topology["bonds"], "bond", np
    )
    angles = _geometry_comparison(
        aa_frames, cg_frames, topology["angles"], "angle", np
    )

    aa_stats = _trajectory_stats(aa_frames, np)
    cg_stats = _trajectory_stats(cg_frames, np)
    volume_delta = _relative_difference(aa_stats["volume_mean"], cg_stats["volume_mean"])
    density_delta = _relative_difference(
        aa_stats["number_density_mean"], cg_stats["number_density_mean"]
    )
    simulation_data = cg_path.parent / "simulation_data.out"
    aa_simulation_data = Path(aa_properties_path) if aa_properties_path else None
    aa_properties = (
        _property_stats(aa_simulation_data)
        if aa_simulation_data is not None
        else []
    )
    cg_properties = _property_stats(simulation_data)
    property_comparison = _property_comparison(aa_properties, cg_properties)

    merged_result = merged_candidate is not None and cg_path == Path(merged_candidate)
    optimized_result = cg_path == best_directory(root / 'optimization') / 'cg.lammpstrj'
    metrics = {
        "format_version": 2,
        "scope": (
            "Mapped-AA versus merged-force-field CG trajectory"
            if merged_result
            else ("Mapped-AA versus per-system optimized CG trajectory" if optimized_result
                  else "Mapped-AA versus baseline CG trajectory")
        ),
        "sources": {
            "mapped_aa": str(aa_path.resolve()),
            "cg": str(cg_path.resolve()),
            "cg_result": ("merged optimization" if merged_result else
                          "per-system optimization" if optimized_result else "baseline simulation"),
            "topology": str(structure_path.resolve()),
            "cg_properties": str(simulation_data.resolve()) if simulation_data.is_file() else "",
            "mapped_aa_properties": (
                str(aa_simulation_data.resolve())
                if aa_simulation_data is not None and aa_simulation_data.is_file()
                else ""
            ),
        },
        "sampling": {
            "mapped_aa_frames": len(aa_frames),
            "cg_frames": len(cg_frames),
            "region": "evenly spaced frames across the remaining trajectory",
            "method": "uniform_frame_indices",
            "maximum_frames_per_trajectory": sampling['maximum_frames'],
            "discard_percent": sampling['discard_percent'],
            "mapped_aa": aa_sampling,
            "cg": cg_sampling,
            "properties": "all records in each property file; trajectory discard does not apply",
        },
        "summary": {
            "rdf_distance": rdf["distance"],
            "bond_distance": bonds.get("distance"),
            "angle_distance": angles.get("distance"),
            "volume_relative_difference": volume_delta,
            "number_density_relative_difference": density_delta,
        },
        "curves": {"rdf": rdf, "bonds": bonds, "angles": angles},
        "topology": {
            "atoms": len(topology["molecules"]),
            "bonds": len(topology["bonds"]),
            "angles": len(topology["angles"]),
            "dihedrals": len(topology["dihedrals"]),
        },
        "trajectory_stats": {"mapped_aa": aa_stats, "cg": cg_stats},
        "property_comparison": property_comparison,
        "cg_properties": cg_properties,
        "notes": [
            "RDF is the all-bead intermolecular radial distribution; same-molecule pairs are excluded.",
            "Distribution distances are total-variation distances from 0 (matching) to 1 (no overlap).",
            "Structural checks sample evenly spaced frames, after the optional initial discard, independently for AA and CG.",
            "Sampling is by frame index, not physical time; it does not establish equilibration or independent samples.",
            "AA properties are loaded from the user-selected LAMMPS ave/time-style output file.",
            "Property statistics use all records in their files, independently of trajectory sampling settings.",
        ],
    }
    destination = root / "validation" / "metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Validation metrics written to: {destination}")
    print(
        "Distances: "
        f"RDF={_display(rdf['distance'])} bond={_display(bonds.get('distance'))} "
        f"angle={_display(angles.get('distance'))}"
    )
    return metrics


def _read_sampled_frames(path: Path, settings=None):
    """Index once, then decode only selected frames; never load the full dump.

    Binary offsets work with both LF and Windows CRLF files. The index pass
    validates frame headers and completeness without splitting every atom row.
    Sampled coordinates use compact NumPy arrays instead of nested Python lists.
    """
    settings = normalize_sampling(settings)
    initial_stat = path.stat()
    signature = (initial_stat.st_size, initial_stat.st_mtime_ns)
    offsets = []
    with path.open('rb') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip() != b"ITEM: TIMESTEP":
                continue
            timestep, atom_count, _, _ = _frame_header(handle, path)
            for _ in range(atom_count):
                row = handle.readline()
                if not row or row.isspace() or row.startswith(b'ITEM:'):
                    raise ValueError(f"Incomplete atom row in {path} at step {timestep}")
            offsets.append(offset)
        if not offsets:
            raise ValueError(f"No complete LAMMPS frames found in {path}")
        indices = sampled_indices(len(offsets), settings)
        frames = []
        for index in indices:
            handle.seek(offsets[index])
            if handle.readline().strip() != b'ITEM: TIMESTEP':
                raise ValueError(f'Trajectory changed while being sampled: {path}')
            frames.append(_read_sampled_frame(handle, path))
    final_stat = path.stat()
    if signature != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError(f'Trajectory changed while being sampled; finish the simulation first: {path}')
    return frames, {
        'total_frames': len(offsets),
        'discarded_frames': indices[0],
        'available_frames': len(offsets) - indices[0],
        'sampled_frames': len(frames),
        'frame_indices': indices,
        'first_timestep': frames[0]['timestep'],
        'last_timestep': frames[-1]['timestep'],
    }


def _frame_header(handle, path):
    """Read a binary dump header, immediately after ITEM: TIMESTEP."""
    timestep_line = handle.readline()
    if not timestep_line:
        raise ValueError(f'Incomplete timestep in {path}')
    timestep = int(float(timestep_line.strip()))
    if handle.readline().strip() != b'ITEM: NUMBER OF ATOMS':
        raise ValueError(f'Malformed atom-count header in {path} at step {timestep}')
    atom_count = int(handle.readline().strip())
    if atom_count < 1:
        raise ValueError(f'No atoms in {path} at step {timestep}')
    if not handle.readline().strip().startswith(b'ITEM: BOX BOUNDS'):
        raise ValueError(f'Malformed box header in {path} at step {timestep}')
    bounds = []
    for _ in range(3):
        parts = handle.readline().split()
        if len(parts) < 2:
            raise ValueError(f'Malformed box bounds in {path} at step {timestep}')
        low, high = float(parts[0]), float(parts[1])
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError(f'Invalid box bounds in {path} at step {timestep}')
        bounds.append((low, high))
    atoms_header = handle.readline().decode('utf-8', errors='replace').strip()
    if not atoms_header.startswith('ITEM: ATOMS '):
        raise ValueError(f'Malformed atom header in {path} at step {timestep}')
    columns = atoms_header.split()[2:]
    lookup = {name: index for index, name in enumerate(columns)}
    if 'id' not in lookup:
        raise ValueError(f'Trajectory requires an id column: {path}')
    _coordinate_columns(lookup)
    return timestep, atom_count, bounds, columns


def _read_sampled_frame(handle, path):
    import numpy as np

    timestep, atom_count, bounds, columns = _frame_header(handle, path)
    lookup = {name: index for index, name in enumerate(columns)}
    coordinate_names = _coordinate_columns(lookup)
    lengths = [high - low for low, high in bounds]
    ids, molecules = [], []
    positions = np.empty((atom_count, 3), dtype=float)
    for row in range(atom_count):
        values = handle.readline().split()
        if len(values) < len(columns):
            raise ValueError(f'Incomplete atom row in {path} at step {timestep}')
        ids.append(int(float(values[lookup['id']])))
        molecules.append(int(float(values[lookup['mol']])) if 'mol' in lookup else 0)
        for axis, name in enumerate(coordinate_names):
            value = float(values[lookup[name]])
            if name in {'xs', 'ys', 'zs'}:
                value = bounds[axis][0] + value * lengths[axis]
            positions[row, axis] = value
    if not np.isfinite(positions).all():
        raise ValueError(f'Non-finite coordinates in {path} at step {timestep}')
    order = sorted(range(atom_count), key=ids.__getitem__)
    return {
        'timestep': timestep,
        'ids': [ids[index] for index in order],
        'molecules': [molecules[index] for index in order],
        'positions': positions[order],
        'box': lengths,
    }


def _coordinate_columns(lookup: dict[str, int]) -> tuple[str, str, str]:
    for names in (("xu", "yu", "zu"), ("x", "y", "z"), ("xs", "ys", "zs")):
        if all(name in lookup for name in names):
            return names
    raise ValueError("Trajectory requires x/y/z, xu/yu/zu, or xs/ys/zs coordinates.")


def _read_topology(path: Path) -> dict:
    atoms = _read_section(path, "Atoms")
    if not atoms:
        raise ValueError(f"No Atoms section found in {path}")
    molecule_by_id = {}
    for row in atoms:
        atom_id = int(row[0])
        molecule_by_id[atom_id] = int(row[1]) if len(row) >= 6 else 0
    atom_ids = sorted(molecule_by_id)
    bonds = [tuple(int(value) for value in row[2:4]) for row in _read_section(path, "Bonds")]
    if not any(molecule_by_id.values()):
        molecule_by_id = _molecule_ids_from_bonds(atom_ids, bonds)
    return {
        "ids": atom_ids,
        "molecules": [molecule_by_id[atom_id] for atom_id in atom_ids],
        "bonds": bonds,
        "angles": [tuple(int(value) for value in row[2:5]) for row in _read_section(path, "Angles")],
        "dihedrals": [tuple(int(value) for value in row[2:6]) for row in _read_section(path, "Dihedrals")],
    }


def _read_section(path: Path, section: str) -> list[list[float]]:
    headers = {"Masses", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers", "Velocities"}
    rows = []
    active = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if active and line in headers:
                break
            if active:
                try:
                    rows.append([float(value) for value in line.split()])
                except ValueError as exc:
                    raise ValueError(f"Invalid {section} row in {path}: {line}") from exc
            elif line == section:
                active = True
    return rows


def _molecule_ids_from_bonds(atom_ids: list[int], bonds: list[tuple[int, int]]) -> dict[int, int]:
    neighbours = {atom_id: set() for atom_id in atom_ids}
    for first, second in bonds:
        if first in neighbours and second in neighbours:
            neighbours[first].add(second)
            neighbours[second].add(first)
    result = {}
    molecule_id = 0
    for start in atom_ids:
        if start in result:
            continue
        molecule_id += 1
        stack = [start]
        result[start] = molecule_id
        while stack:
            current = stack.pop()
            for neighbour in neighbours[current]:
                if neighbour not in result:
                    result[neighbour] = molecule_id
                    stack.append(neighbour)
    return result


def _check_compatible(aa_frames: list[dict], cg_frames: list[dict], topology: dict) -> None:
    expected = topology["ids"]
    for label, frames in (("mapped-AA", aa_frames), ("CG", cg_frames)):
        first = frames[0]["ids"]
        if first != expected:
            missing = sorted(set(expected) - set(first))[:8]
            extra = sorted(set(first) - set(expected))[:8]
            raise ValueError(
                f"{label} atom IDs do not match structure.data; missing={missing}, extra={extra}."
            )
        if any(frame["ids"] != first for frame in frames[1:]):
            raise ValueError(f"{label} atom IDs change between sampled frames.")


def _rdf_comparison(aa_frames, cg_frames, topology_molecules, np) -> dict:
    count = len(topology_molecules)
    if count < 2:
        return _empty_curve("Not enough beads for an RDF.")
    selected = np.linspace(0, count - 1, min(count, MAX_RDF_ATOMS), dtype=int)
    selected = np.unique(selected)
    molecule_ids = np.asarray(topology_molecules, dtype=int)[selected]
    minimum_box = min(min(frame["box"]) for frame in [*aa_frames, *cg_frames])
    cutoff = max(0.1, 0.49 * minimum_box)
    edges = np.linspace(0.0, cutoff, 101)
    aa = _rdf_for_frames(aa_frames, selected, molecule_ids, edges, np)
    cg = _rdf_for_frames(cg_frames, selected, molecule_ids, edges, np)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "x": centers.tolist(),
        "mapped_aa": aa.tolist(),
        "cg": cg.tolist(),
        "distance": _distribution_distance(aa, cg, np),
        "x_label": "r (Å)",
        "y_label": "g(r)",
        "detail": f"{len(selected)} beads sampled; same-molecule pairs excluded",
    }


def _rdf_for_frames(frames, selected, molecule_ids, edges, np):
    observed = np.zeros(len(edges) - 1, dtype=float)
    expected = np.zeros(len(edges) - 1, dtype=float)
    upper = np.triu_indices(len(selected), 1)
    allowed = molecule_ids[upper[0]] != molecule_ids[upper[1]]
    allowed_pairs = int(np.count_nonzero(allowed))
    first, second = upper[0][allowed], upper[1][allowed]
    shell_volumes = 4.0 * math.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    for frame in frames:
        positions = np.asarray(frame["positions"], dtype=float)[selected]
        box = np.asarray(frame["box"], dtype=float)
        delta = positions[first] - positions[second]
        delta -= np.rint(delta / box) * box
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        observed += np.histogram(distances, bins=edges)[0]
        expected += allowed_pairs * shell_volumes / float(np.prod(box))
    return np.divide(observed, expected, out=np.zeros_like(observed), where=expected > 0.0)


def _geometry_comparison(aa_frames, cg_frames, interactions, kind, np) -> dict:
    if not interactions:
        return _empty_curve(f"No {kind} interactions are present at this interaction level.")
    id_to_index = {atom_id: index for index, atom_id in enumerate(aa_frames[0]["ids"])}
    try:
        indexed = np.asarray(
            [[id_to_index[atom_id] for atom_id in interaction] for interaction in interactions],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError(f"Topology {kind} references unknown atom ID {exc.args[0]}.") from exc
    aa_values = _geometry_values(aa_frames, indexed, kind, np)
    cg_values = _geometry_values(cg_frames, indexed, kind, np)
    if kind == "angle":
        edges = np.linspace(0.0, 180.0, 91)
        x_label = "angle (degrees)"
    else:
        maximum = max(float(np.max(np.concatenate((aa_values, cg_values)))) * 1.05, 0.25)
        edges = np.linspace(0.0, maximum, 91)
        x_label = "distance (Å)"
    aa_hist = np.histogram(aa_values, bins=edges)[0].astype(float)
    cg_hist = np.histogram(cg_values, bins=edges)[0].astype(float)
    aa_hist /= max(float(aa_hist.sum()), 1.0)
    cg_hist /= max(float(cg_hist.sum()), 1.0)
    return {
        "x": (0.5 * (edges[:-1] + edges[1:])).tolist(),
        "mapped_aa": aa_hist.tolist(),
        "cg": cg_hist.tolist(),
        "distance": _distribution_distance(aa_hist, cg_hist, np),
        "x_label": x_label,
        "y_label": "probability",
        "detail": f"{len(interactions)} {kind} interactions",
        "mapped_aa_mean": float(np.mean(aa_values)),
        "mapped_aa_std": float(np.std(aa_values)),
        "cg_mean": float(np.mean(cg_values)),
        "cg_std": float(np.std(cg_values)),
    }


def _geometry_values(frames, interactions, kind, np):
    values = []
    for frame in frames:
        positions = np.asarray(frame["positions"], dtype=float)
        box = np.asarray(frame["box"], dtype=float)
        if kind == "bond":
            delta = positions[interactions[:, 1]] - positions[interactions[:, 0]]
            delta -= np.rint(delta / box) * box
            values.append(np.sqrt(np.sum(delta * delta, axis=1)))
        else:
            first = positions[interactions[:, 0]] - positions[interactions[:, 1]]
            second = positions[interactions[:, 2]] - positions[interactions[:, 1]]
            first -= np.rint(first / box) * box
            second -= np.rint(second / box) * box
            denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
            cosine = np.divide(
                np.sum(first * second, axis=1),
                denominator,
                out=np.ones_like(denominator),
                where=denominator > 0.0,
            )
            values.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return np.concatenate(values)


def _distribution_distance(first, second, np) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first / max(float(first.sum()), 1.0e-15)
    second = second / max(float(second.sum()), 1.0e-15)
    return float(0.5 * np.sum(np.abs(first - second)))


def _trajectory_stats(frames, np) -> dict:
    volumes = np.asarray([np.prod(frame["box"]) for frame in frames], dtype=float)
    atom_count = len(frames[0]["ids"])
    number_density = atom_count / volumes
    return {
        "sampled_frames": len(frames),
        "first_timestep": frames[0]["timestep"],
        "last_timestep": frames[-1]["timestep"],
        "atoms": atom_count,
        "volume_mean": float(np.mean(volumes)),
        "volume_std": float(np.std(volumes)),
        "number_density_mean": float(np.mean(number_density)),
        "number_density_std": float(np.std(number_density)),
    }


def load_property_statistics(
    path: str | Path,
    requested_properties: list[dict] | None = None,
) -> list[dict]:
    """Read matching property columns from a LAMMPS ave/time-style table."""
    source = Path(path)
    columns, rows = _read_ave_time_table(source)
    requested = requested_properties or PROPERTY_COLUMNS
    normalized_columns = {_normalized_column(name): index for index, name in enumerate(columns)}
    result = []
    for definition in requested:
        label = str(definition.get("label") or definition.get("column") or "Property")
        canonical = str(definition.get("column") or "")
        candidates = PROPERTY_ALIASES.get(canonical, (canonical,))
        column_index = next(
            (
                normalized_columns[_normalized_column(candidate)]
                for candidate in candidates
                if _normalized_column(candidate) in normalized_columns
            ),
            None,
        )
        if column_index is None:
            continue
        values = [row[column_index] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        result.append(
            {
                "label": label,
                "column": canonical,
                "source_column": columns[column_index],
                "mean": mean,
                "std": math.sqrt(variance),
                "minimum": min(values),
                "maximum": max(values),
            }
        )
    return result


def _read_ave_time_table(path: Path) -> tuple[list[str], list[list[float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Properties file not found: {path}")
    header_candidates = []
    rows = []
    width = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                candidate = line.lstrip("#").strip().split()
                if candidate:
                    header_candidates.append(candidate)
                continue
            parts = line.split()
            try:
                values = [float(value) for value in parts]
            except ValueError:
                header_candidates.append(parts)
                continue
            if width is None:
                width = len(values)
            elif len(values) != width:
                raise ValueError(
                    f"Inconsistent column count in {path}: expected {width}, found {len(values)}."
                )
            rows.append(values)
    if not rows or width is None:
        raise ValueError(f"No numeric ave/time rows found in {path}.")

    header = next(
        (candidate for candidate in reversed(header_candidates) if len(candidate) == width),
        None,
    )
    if header is None:
        from .corrector.lammps_reader import SIMULATION_DATA_COLUMNS

        if width == len(SIMULATION_DATA_COLUMNS):
            header = list(SIMULATION_DATA_COLUMNS)
        else:
            raise ValueError(
                "Could not identify property columns. Include a whitespace-separated header "
                "such as '# TimeStep temperature pressure density'."
            )
    return header, rows


def _normalized_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _property_stats(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        return load_property_statistics(path)
    except (OSError, ValueError) as exc:
        print(f"Property statistics skipped: {exc}")
        return []


def _property_comparison(
    aa_properties: list[dict],
    cg_properties: list[dict],
) -> list[dict]:
    """Return CG properties in order, restricted to common columns once AA is loaded."""
    aa_by_column = {item["column"]: item for item in aa_properties}
    rows = []
    for cg_item in cg_properties:
        column = cg_item["column"]
        aa_item = aa_by_column.get(column)
        if aa_properties and aa_item is None:
            continue
        rows.append(
            {
                "label": cg_item["label"],
                "column": column,
                "mapped_aa": _mean_std(aa_item),
                "cg": _mean_std(cg_item),
            }
        )
    return rows


def _mean_std(item: dict | None) -> dict:
    if not item:
        return {"mean": None, "std": None}
    return {"mean": item.get("mean"), "std": item.get("std")}


def _relative_difference(reference: float, candidate: float) -> float | None:
    if abs(reference) < 1.0e-15:
        return None
    return abs(candidate - reference) / abs(reference)


def _empty_curve(detail: str) -> dict:
    return {
        "x": [],
        "mapped_aa": [],
        "cg": [],
        "distance": None,
        "x_label": "",
        "y_label": "",
        "detail": detail,
    }


def _display(value) -> str:
    return "n/a" if value is None else f"{value:.4g}"
