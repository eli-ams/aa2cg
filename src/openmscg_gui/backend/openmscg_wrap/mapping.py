from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from .chemistry import molecular_formula_to_atoms, molecular_mass
from .config import MoleculeConfig


@dataclass
class Atom:
    atom_id: int
    element: str


@dataclass
class Bead:
    name: str
    molecule_index: int
    local_index: int
    formula: str
    bead_type: str
    atom_ids: list[int]


@dataclass
class Molecule:
    name: str
    number: int
    formula: str
    beads_per_molecule: int
    beads: list[Bead]
    bead_connections: list[tuple[int, int]] = field(default_factory=list)


def read_ndx(path: Path) -> list[Bead]:
    beads: list[Bead] = []
    current_name: str | None = None
    current_atoms: list[int] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            if current_name is not None:
                beads.append(_make_bead(current_name, current_atoms))
            current_name = line[1:-1].strip()
            current_atoms = []
        else:
            current_atoms.extend(int(value) for value in line.split())

    if current_name is not None:
        beads.append(_make_bead(current_name, current_atoms))
    return beads


def _make_bead(name: str, atom_ids: list[int]) -> Bead:
    parts = name.split("_")
    if len(parts) != 4:
        raise ValueError(
            "Bead names must use MoleculeID_BeadID_Formula_Type, "
            f"got {name!r}."
        )
    return Bead(
        name=name,
        molecule_index=int(parts[0]),
        local_index=int(parts[1]),
        formula=parts[2],
        bead_type=parts[3],
        atom_ids=atom_ids,
    )


def read_pdb(path: Path) -> tuple[list[Atom], list[list[int]]]:
    atoms: list[Atom] = []
    bonds: list[list[int]] = []
    for line in path.read_text().splitlines():
        if line.startswith("HETATM") or line.startswith("ATOM"):
            parts = line.split()
            element = line[76:78].strip() if len(line) >= 78 else ""
            atoms.append(Atom(atom_id=int(parts[1]), element=element or _element_from_pdb_name(parts[2])))
        elif line.startswith("CONECT"):
            values = [int(value) for value in line.split()[1:]]
            if len(values) > 1:
                bonds.append(values)
    return atoms, bonds


def _element_from_pdb_name(name: str) -> str:
    match = re.match(r"[A-Za-z]+", name)
    if not match:
        raise ValueError(f"Could not infer element from PDB atom name: {name}")
    value = match.group()
    return value[0] if len(value) > 1 else value


def build_molecules(configs: list[MoleculeConfig], beads: list[Bead]) -> list[Molecule]:
    molecules: list[Molecule] = []
    offset = 0
    for index, config in enumerate(configs, start=1):
        if config.formula is None or config.beads_per_molecule is None:
            raise ValueError(
                f"Molecule {config.name!r} needs formula and beads_per_molecule. "
                "For SMILES input, run automatic preparation first."
            )
        next_offset = offset + config.beads_per_molecule
        molecule_beads = beads[offset:next_offset]
        if len(molecule_beads) != config.beads_per_molecule:
            raise ValueError(f"Expected {config.beads_per_molecule} beads for {config.name}.")
        for expected_local, bead in enumerate(molecule_beads, start=1):
            if bead.molecule_index != index or bead.local_index != expected_local:
                raise ValueError(f"Unexpected bead order/name in NDX: {bead.name}")
        molecules.append(
            Molecule(
                name=config.name,
                number=config.number,
                formula=config.formula,
                beads_per_molecule=config.beads_per_molecule,
                beads=molecule_beads,
            )
        )
        offset = next_offset
    if offset != len(beads):
        raise ValueError("NDX contains extra beads not described by the config.")
    return molecules


def validate_bead_formulae(molecules: list[Molecule]) -> None:
    for molecule in molecules:
        expected = sorted(molecular_formula_to_atoms(molecule.formula))
        actual = sorted(atom for bead in molecule.beads for atom in molecular_formula_to_atoms(bead.formula))
        if expected != actual:
            raise ValueError(
                f"Formula mismatch for {molecule.name}: config {molecule.formula}, "
                f"beads {''.join(bead.formula for bead in molecule.beads)}"
            )


def add_bead_connections(molecules: list[Molecule], pdb_bonds: list[list[int]]) -> None:
    atom_to_bead: dict[int, tuple[Molecule, int]] = {}
    for molecule in molecules:
        for bead_index, bead in enumerate(molecule.beads):
            for atom_id in bead.atom_ids:
                atom_to_bead[atom_id] = (molecule, bead_index)

    connections_by_molecule: dict[int, set[tuple[int, int]]] = {id(mol): set() for mol in molecules}
    for row in pdb_bonds:
        parent = row[0]
        for child in row[1:]:
            if parent not in atom_to_bead or child not in atom_to_bead:
                continue
            mol_a, bead_a = atom_to_bead[parent]
            mol_b, bead_b = atom_to_bead[child]
            if mol_a is not mol_b or bead_a == bead_b:
                continue
            connections_by_molecule[id(mol_a)].add(tuple(sorted((bead_a, bead_b))))

    for molecule in molecules:
        molecule.bead_connections = sorted(connections_by_molecule[id(molecule)])


def unique_bead_types(molecules: list[Molecule]) -> list[str]:
    values: list[str] = []
    for molecule in molecules:
        for bead in molecule.beads:
            if bead.bead_type not in values:
                values.append(bead.bead_type)
    return values


def bead_type_records(molecules: list[Molecule], bead_types: list[str]) -> list[tuple[int, str, int]]:
    records: list[tuple[int, str, int]] = []
    counter = 1
    for molecule in molecules:
        for bead in molecule.beads:
            records.append((counter, bead.bead_type, bead_types.index(bead.bead_type)))
            counter += 1
    return records


def write_map_yaml(path: Path, molecules: list[Molecule], atoms: list[Atom]) -> None:
    atom_lookup = {atom.atom_id: atom.element for atom in atoms}
    lines: list[str] = ["site-types:"]
    for molecule in molecules:
        atom_offset = min(atom_id for bead in molecule.beads for atom_id in bead.atom_ids)
        for bead in molecule.beads:
            elements = [atom_lookup[atom_id] for atom_id in bead.atom_ids]
            lines.append(f"  {bead.name}:")
            lines.append(f"    index: {[atom_id - atom_offset for atom_id in bead.atom_ids]}")
            lines.append(f"    x-weight: {[molecular_mass(element) for element in elements]}")
            lines.append(f"    f-weight: {[1.0 for _ in bead.atom_ids]}")

    lines.append("system:")
    anchor = 0
    for molecule in molecules:
        atoms_per_molecule = sum(len(bead.atom_ids) for bead in molecule.beads)
        lines.append(f"  - anchor: {anchor}")
        lines.append(f"    repeat: {molecule.number}")
        lines.append(f"    offset: {atoms_per_molecule}")
        lines.append("    sites:")
        for bead in molecule.beads:
            lines.append(f"      - [ {bead.name}, 0 ]")
        anchor += molecule.number * atoms_per_molecule

    path.write_text("\n".join(lines) + "\n")


def replace_cg_type_ids(input_path: Path, output_path: Path, records: list[tuple[int, str, int]]) -> None:
    bead_to_type = {bead_id: type_id + 1 for bead_id, _, type_id in records}
    with input_path.open() as source, output_path.open("w") as target:
        for line in source:
            values = _numeric_values(line)
            if values is None:
                target.write(line)
                continue
            values[1] = bead_to_type[int(values[1])]
            target.write(" ".join(_format_number(value) for value in values) + "\n")


def _numeric_values(line: str) -> list[int | float] | None:
    try:
        values: list[int | float] = []
        for value in line.split():
            values.append(float(value) if re.search(r"[.eE]", value) else int(value))
        if len(values) >= 5:
            return values
    except ValueError:
        return None
    return None


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
