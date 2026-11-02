from __future__ import annotations

from pathlib import Path

from .chemistry import molecular_formula_to_atoms
from .mapping import Atom, Molecule


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def validate_pdb_against_config(atoms: list[Atom], molecules: list[Molecule]) -> None:
    expected_atoms_per_unit = sum(len(molecular_formula_to_atoms(_require_formula(molecule))) for molecule in molecules)
    pdb_atoms_per_unit = len(atoms)
    if pdb_atoms_per_unit != expected_atoms_per_unit:
        raise ValueError(
            f"PDB atom count ({pdb_atoms_per_unit}) does not match one configured system unit "
            f"({expected_atoms_per_unit})."
        )


def validate_trajectory_atom_count(path: Path, molecules: list[Molecule]) -> None:
    expected = sum(
        molecule.number * len(molecular_formula_to_atoms(_require_formula(molecule)))
        for molecule in molecules
    )
    with path.open() as handle:
        for index, line in enumerate(handle):
            if index == 3:
                actual = int(line.strip())
                break
        else:
            raise ValueError(f"Could not read atom count from {path}.")
    if actual != expected:
        raise ValueError(
            f"Trajectory atom count ({actual}) does not match configured system ({expected})."
        )


def validate_trajectory_atom_ids(path: Path, expected_atom_count: int) -> None:
    from .trajectory_mapping import read_first_lammps_frame

    frame = read_first_lammps_frame(path)
    if "id" not in frame["columns"]:
        raise ValueError(
            "Topology-driven mapping requires an 'id' column in the LAMMPS trajectory."
        )
    atom_ids = sorted(int(row["id"]) for row in frame["rows"])
    expected = list(range(1, expected_atom_count + 1))
    if atom_ids != expected:
        raise ValueError(
            "Trajectory atom IDs do not match structure.data. Expected contiguous IDs "
            f"1..{expected_atom_count}."
        )


def _require_formula(molecule: Molecule) -> str:
    if molecule.formula is None:
        raise ValueError(f"Molecule {molecule.name} is missing a molecular formula.")
    return molecule.formula
