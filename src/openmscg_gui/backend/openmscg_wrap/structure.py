from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math
import os
import shutil

from .chemistry import molecular_mass
from .mapping import Molecule, unique_bead_types
from .topology import infer_interaction_level, pair_interactions


def generate_structure_data(
    topology_file: Path,
    trajectory_file: Path,
    output_path: Path,
    title: str,
    molecules: list[Molecule],
    interaction_level: str | None = None,
) -> None:
    import numpy as np
    import mscg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    level = interaction_level or infer_interaction_level(molecules)
    bead_types = unique_bead_types(molecules)
    masses = masses_for_bead_types(molecules, bead_types)

    trajectory = mscg.Trajectory(str(trajectory_file), fmt="lammpstrj")
    trajectory.read_frame()
    topology = mscg.Topology.read_file(str(topology_file))
    topology.system_name = title

    save_kwargs = {
        "masses": masses,
        "box": np.vstack([np.zeros(3), trajectory.box]).T,
        "x": trajectory.x,
    }
    if level != "Pair":
        save_kwargs["molecule"] = molecule_ids(molecules)

    with pushd(output_path.parent):
        topology.save("lammps", **save_kwargs)
        generated = output_path.parent / f"{title}.data"
        if generated != output_path:
            if output_path.exists():
                output_path.unlink()
            shutil.move(str(generated), str(output_path))


def interactions_from_structure_data(path: Path) -> dict[str, list[tuple[str, ...]]]:
    masses = mass_types(path)
    atom_types_by_atom_id = atom_types(path, masses)
    interactions = {
        "Pair": [tuple(item) for item in pair_interactions([name for _, name in masses])],
        "Bond": bond_types(path, atom_types_by_atom_id),
        "Angle": angle_types(path, atom_types_by_atom_id),
        "Dihedral": dihedral_types(path, atom_types_by_atom_id),
    }
    return {kind: values for kind, values in interactions.items() if values}


def validate_structure_bonds(path: Path, max_bond_length: float = 3.0) -> dict[str, float | int]:
    """Reject a LAMMPS structure whose bonds do not match its coordinates."""
    atom_rows = read_section(path, "Atoms")
    bond_rows = read_section(path, "Bonds")
    if not atom_rows:
        raise ValueError(f"No Atoms section found in {path}.")
    if not bond_rows:
        return {"atoms": len(atom_rows), "bonds": 0, "max_bond_length": 0.0}

    atoms = {}
    for row in atom_rows:
        atom_id = int(row[0])
        if len(row) < 6:
            raise ValueError(f"Bonded structure uses an incompatible Atoms row in {path}: {row}")
        atoms[atom_id] = {
            "molecule": int(row[1]),
            "position": tuple(float(value) for value in row[-3:]),
        }
    box = _box_lengths(path)
    distances = []
    invalid = []
    for row in bond_rows:
        first, second = int(row[2]), int(row[3])
        if first not in atoms or second not in atoms:
            raise ValueError(f"Bond references a missing atom in {path}: {row}")
        if atoms[first]["molecule"] != atoms[second]["molecule"]:
            invalid.append((first, second, float("inf"), "different molecules"))
            continue
        delta = [
            atoms[second]["position"][axis] - atoms[first]["position"][axis]
            for axis in range(3)
        ]
        for axis, length in enumerate(box):
            if length > 0.0:
                delta[axis] -= round(delta[axis] / length) * length
        distance = math.sqrt(sum(value * value for value in delta))
        distances.append(distance)
        if distance > max_bond_length:
            invalid.append((first, second, distance, "too long"))

    if invalid:
        examples = ", ".join(
            f"{first}-{second} ({reason}, {distance:.3f} Å)"
            for first, second, distance, reason in invalid[:8]
        )
        raise ValueError(
            "Generated structure.data does not align topology bonds with mapped coordinates. "
            f"Found {len(invalid)} invalid bonds above {max_bond_length:.2f} Å; examples: {examples}. "
            "Check the molecule/trajectory atom mapping before running force matching or LAMMPS."
        )
    return {
        "atoms": len(atom_rows),
        "bonds": len(bond_rows),
        "max_bond_length": max(distances, default=0.0),
    }


def _box_lengths(path: Path) -> tuple[float, float, float]:
    values = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-2:] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
            values[parts[-2][0]] = float(parts[1]) - float(parts[0])
    if set(values) != {"x", "y", "z"}:
        raise ValueError(f"Could not read orthogonal box bounds from {path}.")
    return values["x"], values["y"], values["z"]


def mass_types(path: Path) -> list[tuple[int, str]]:
    rows = read_section(path, "Masses")
    if not rows:
        raise ValueError(f"No Masses section found in {path}.")
    return [(int(row[0]), str(row[-1])) for row in rows]


def atom_types(path: Path, masses: list[tuple[int, str]]) -> dict[int, str]:
    rows = read_section(path, "Atoms")
    if not rows:
        raise ValueError(f"No Atoms section found in {path}.")
    mass_lookup = {type_id: name for type_id, name in masses}
    result: dict[int, str] = {}
    for row in rows:
        atom_id = int(row[0])
        type_id = int(row[1]) if len(row) == 5 else int(row[2])
        result[atom_id] = mass_lookup[type_id]
    return result


def bond_types(path: Path, atom_types_by_atom_id: dict[int, str]) -> list[tuple[str, str]]:
    rows = read_section(path, "Bonds")
    by_type_id: dict[int, tuple[int, int]] = {}
    for row in rows:
        type_id = int(row[1])
        by_type_id.setdefault(type_id, (int(row[2]), int(row[3])))
    return [
        (atom_types_by_atom_id[a], atom_types_by_atom_id[b])
        for _, (a, b) in sorted(by_type_id.items())
    ]


def angle_types(path: Path, atom_types_by_atom_id: dict[int, str]) -> list[tuple[str, str, str]]:
    rows = read_section(path, "Angles")
    by_type_id: dict[int, tuple[int, int, int]] = {}
    for row in rows:
        type_id = int(row[1])
        by_type_id.setdefault(type_id, (int(row[2]), int(row[3]), int(row[4])))
    return [
        (atom_types_by_atom_id[a], atom_types_by_atom_id[b], atom_types_by_atom_id[c])
        for _, (a, b, c) in sorted(by_type_id.items())
    ]


def dihedral_types(path: Path, atom_types_by_atom_id: dict[int, str]) -> list[tuple[str, str, str, str]]:
    rows = read_section(path, "Dihedrals")
    by_type_id: dict[int, tuple[int, int, int, int]] = {}
    for row in rows:
        type_id = int(row[1])
        by_type_id.setdefault(type_id, (int(row[2]), int(row[3]), int(row[4]), int(row[5])))
    return [
        (atom_types_by_atom_id[a], atom_types_by_atom_id[b], atom_types_by_atom_id[c], atom_types_by_atom_id[d])
        for _, (a, b, c, d) in sorted(by_type_id.items())
    ]


def read_section(path: Path, section: str) -> list[list[int | float | str]]:
    rows: list[list[int | float | str]] = []
    in_section = False
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip() if section == "Masses" else raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if in_section:
                if _is_section_header(line):
                    break
                rows.append([_number_or_text(value) for value in line.split()])
            elif line == section:
                in_section = True
    return rows


def masses_for_bead_types(molecules: list[Molecule], bead_types: list[str]) -> list[float]:
    masses: list[float] = []
    for bead_type in bead_types:
        bead = next(
            bead
            for molecule in molecules
            for bead in molecule.beads
            if bead.bead_type == bead_type
        )
        masses.append(molecular_mass(bead.formula))
    return masses


def molecule_ids(molecules: list[Molecule]) -> list[int]:
    ids: list[int] = []
    counter = 0
    for molecule in molecules:
        for _ in range(molecule.number):
            counter += 1
            ids.extend([counter] * molecule.beads_per_molecule)
    return ids


def _is_section_header(line: str) -> bool:
    return line in {
        "Masses",
        "Atoms",
        "Bonds",
        "Angles",
        "Dihedrals",
        "Velocities",
        "Impropers",
    }


def _number_or_text(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(previous))
