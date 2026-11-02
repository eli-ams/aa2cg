from __future__ import annotations

from pathlib import Path

from .mapping import Molecule, unique_bead_types


INTERACTION_LEVEL_ORDER = {"Pair": 1, "Bond": 1, "Angle": 2, "Dihedral": 3}


def infer_interaction_level(molecules: list[Molecule]) -> str:
    max_beads = max(min(molecule.beads_per_molecule, 4) for molecule in molecules)
    return {1: "Pair", 2: "Bond", 3: "Angle", 4: "Dihedral"}[max_beads]


def write_topology(path: Path, molecules: list[Molecule], interaction_level: str | None = None) -> None:
    bead_types = unique_bead_types(molecules)
    level = interaction_level or infer_interaction_level(molecules)
    if level not in INTERACTION_LEVEL_ORDER:
        raise ValueError(f"Unsupported interaction level: {level}")

    lines: list[str] = []
    lines.append(f"cgsites {sum(mol.number * mol.beads_per_molecule for mol in molecules)}")
    lines.append(f"cgtypes {len(bead_types)}")
    lines.extend(bead_types)
    lines.append(f"moltypes {len(molecules)}")
    for molecule in molecules:
        lines.append(f"mol {molecule.beads_per_molecule} {INTERACTION_LEVEL_ORDER[level]}")
        lines.append("sitetypes")
        lines.extend(bead.bead_type for bead in molecule.beads)
        lines.append(f"bonds {len(molecule.bead_connections)}")
        lines.extend(f"{a + 1} {b + 1}" for a, b in molecule.bead_connections)
    lines.append(f"system {len(molecules)}")
    lines.extend(f"{index} {molecule.number}" for index, molecule in enumerate(molecules, start=1))
    path.write_text("\n".join(lines) + "\n")


def pair_interactions(bead_types: list[str]) -> list[tuple[str, str]]:
    return [(bead_types[i], bead_types[j]) for i in range(len(bead_types)) for j in range(i, len(bead_types))]
