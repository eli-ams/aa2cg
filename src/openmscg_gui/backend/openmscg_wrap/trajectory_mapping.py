from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import math


MASS_TO_ELEMENT = {
    "H": 1.008,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Si": 28.085,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}

COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Si": 1.11,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


def align_explicit_molecule_to_trajectory(
    mol_with_hydrogens,
    trajectory_path: str | Path,
    molecule_count: int,
) -> dict[int, int]:
    """Map RDKit atom indices to per-molecule trajectory positions.

    The LAMMPS dump can list rows in any order. Atoms are grouped by molecule ID,
    sorted by atom ID, unwrapped, and connected geometrically. An element-labelled
    graph isomorphism then aligns the drawn RDKit graph with that trajectory graph.
    """
    frame = read_first_lammps_frame(Path(trajectory_path))
    source_elements = [atom.GetSymbol() for atom in mol_with_hydrogens.GetAtoms()]
    atoms_per_molecule = len(source_elements)
    groups = _molecule_groups(frame, atoms_per_molecule, molecule_count)
    reference = groups[0]
    target_elements = [atom["element"] for atom in reference]
    if Counter(source_elements) != Counter(target_elements):
        raise ValueError(
            "The drawn molecule and first trajectory molecule have different element counts: "
            f"drawing={dict(Counter(source_elements))}, trajectory={dict(Counter(target_elements))}."
        )

    for index, group in enumerate(groups[1:], start=2):
        elements = [atom["element"] for atom in group]
        if elements != target_elements:
            raise ValueError(
                f"Trajectory molecule {index} does not use the same atom ordering/elements "
                "as the first molecule."
            )

    source_adjacency = [set() for _ in source_elements]
    for bond in mol_with_hydrogens.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        source_adjacency[first].add(second)
        source_adjacency[second].add(first)
    target_adjacency = infer_covalent_graph(reference)
    return labelled_graph_isomorphism(
        source_elements,
        source_adjacency,
        target_elements,
        target_adjacency,
    )


def read_first_lammps_frame(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory does not exist: {path}")
    with path.open() as handle:
        line = handle.readline()
        while line and not line.startswith("ITEM: TIMESTEP"):
            line = handle.readline()
        if not line:
            raise ValueError(f"No LAMMPS frame found in {path}.")
        handle.readline()
        if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Malformed LAMMPS frame in {path}: atom-count header missing.")
        atom_count = int(handle.readline().strip())
        bounds_header = handle.readline().strip()
        if not bounds_header.startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Malformed LAMMPS frame in {path}: box bounds missing.")
        bounds = []
        for _ in range(3):
            values = [float(value) for value in handle.readline().split()[:2]]
            bounds.append((values[0], values[1]))
        atoms_header = handle.readline().strip()
        if not atoms_header.startswith("ITEM: ATOMS"):
            raise ValueError(f"Malformed LAMMPS frame in {path}: atom columns missing.")
        columns = atoms_header.split()[2:]
        rows = []
        for _ in range(atom_count):
            values = handle.readline().split()
            if len(values) != len(columns):
                raise ValueError(f"Malformed atom row in {path}.")
            rows.append(dict(zip(columns, values)))
    return {"bounds": bounds, "columns": columns, "rows": rows}


def _molecule_groups(frame: dict, atoms_per_molecule: int, molecule_count: int) -> list[list[dict]]:
    rows = frame["rows"]
    columns = set(frame["columns"])
    if "id" not in columns:
        raise ValueError("Automatic trajectory alignment requires an 'id' column.")
    prepared = [_prepare_atom(row, frame["bounds"], columns) for row in rows]

    groups: list[list[dict]] = []
    if "mol" in columns:
        by_molecule = defaultdict(list)
        for atom in prepared:
            by_molecule[int(atom["mol"])].append(atom)
        for molecule_id in sorted(by_molecule):
            group = sorted(by_molecule[molecule_id], key=lambda atom: atom["id"])
            groups.append(group)
    else:
        ordered = sorted(prepared, key=lambda atom: atom["id"])
        groups = [
            ordered[start : start + atoms_per_molecule]
            for start in range(0, len(ordered), atoms_per_molecule)
        ]

    if len(groups) != molecule_count:
        raise ValueError(
            f"Expected {molecule_count} molecules in the trajectory, found {len(groups)}."
        )
    bad_sizes = [len(group) for group in groups if len(group) != atoms_per_molecule]
    if bad_sizes:
        raise ValueError(
            f"Expected {atoms_per_molecule} atoms per molecule, found incompatible group sizes {bad_sizes[:5]}."
        )
    return groups


def _prepare_atom(row: dict[str, str], bounds, columns: set[str]) -> dict:
    element = row.get("element") or row.get("el")
    if element is None:
        if "mass" not in columns:
            raise ValueError(
                "Automatic trajectory alignment requires either an element or mass column."
            )
        element = _element_from_mass(float(row["mass"]))

    position = []
    for axis, (low, high) in zip("xyz", bounds):
        unwrapped = f"{axis}u"
        scaled = f"{axis}s"
        if unwrapped in columns:
            coordinate = float(row[unwrapped])
        elif axis in columns:
            coordinate = float(row[axis])
            image = f"i{axis}"
            if image in columns:
                coordinate += int(row[image]) * (high - low)
        elif scaled in columns:
            coordinate = low + float(row[scaled]) * (high - low)
        else:
            raise ValueError("Automatic trajectory alignment requires x/y/z coordinates.")
        position.append(coordinate)
    return {
        "id": int(row["id"]),
        "mol": int(row["mol"]) if "mol" in row else None,
        "element": element,
        "position": tuple(position),
    }


def _element_from_mass(mass: float) -> str:
    element, reference = min(MASS_TO_ELEMENT.items(), key=lambda item: abs(item[1] - mass))
    if abs(reference - mass) > 0.75:
        raise ValueError(f"Could not infer an element from trajectory mass {mass}.")
    return element


def infer_covalent_graph(atoms: list[dict]) -> list[set[int]]:
    adjacency = [set() for _ in atoms]
    for first in range(len(atoms)):
        for second in range(first + 1, len(atoms)):
            element_a = atoms[first]["element"]
            element_b = atoms[second]["element"]
            if element_a not in COVALENT_RADII or element_b not in COVALENT_RADII:
                raise ValueError(f"No covalent radius configured for {element_a} or {element_b}.")
            cutoff = 1.25 * (COVALENT_RADII[element_a] + COVALENT_RADII[element_b])
            distance = math.dist(atoms[first]["position"], atoms[second]["position"])
            if 0.35 < distance <= cutoff:
                adjacency[first].add(second)
                adjacency[second].add(first)
    return adjacency


def labelled_graph_isomorphism(
    source_elements: list[str],
    source_adjacency: list[set[int]],
    target_elements: list[str],
    target_adjacency: list[set[int]],
) -> dict[int, int]:
    if len(source_elements) != len(target_elements):
        raise ValueError("Molecular graphs have different atom counts.")
    if sum(map(len, source_adjacency)) != sum(map(len, target_adjacency)):
        raise ValueError(
            "Could not align the drawn molecule to the trajectory: inferred bond count differs "
            f"(drawing={sum(map(len, source_adjacency)) // 2}, "
            f"trajectory={sum(map(len, target_adjacency)) // 2})."
        )

    source_labels = _refined_labels(source_elements, source_adjacency)
    target_labels = _refined_labels(target_elements, target_adjacency)
    if Counter(source_labels) != Counter(target_labels):
        raise ValueError(
            "Could not align the drawn molecule to the trajectory: element/connectivity signatures differ."
        )
    candidates = {
        source: [target for target, label in enumerate(target_labels) if label == source_labels[source]]
        for source in range(len(source_elements))
    }
    order = sorted(
        range(len(source_elements)),
        key=lambda node: (len(candidates[node]), -len(source_adjacency[node]), node),
    )
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def search(position: int) -> bool:
        if position == len(order):
            return True
        source = order[position]
        for target in candidates[source]:
            if target in used:
                continue
            if any(
                ((other in source_adjacency[source]) != (mapped in target_adjacency[target]))
                for other, mapped in mapping.items()
            ):
                continue
            mapping[source] = target
            used.add(target)
            if search(position + 1):
                return True
            used.remove(target)
            del mapping[source]
        return False

    if not search(0):
        raise ValueError("Could not find a graph isomorphism between drawing and trajectory molecule.")
    return mapping


def _refined_labels(elements: list[str], adjacency: list[set[int]]) -> list[tuple]:
    labels: list[object] = [(element, len(adjacency[index])) for index, element in enumerate(elements)]
    for _ in range(min(6, len(elements))):
        raw = [(labels[index], tuple(sorted(labels[neighbor] for neighbor in adjacency[index]))) for index in range(len(elements))]
        unique = {value: rank for rank, value in enumerate(sorted(set(raw), key=repr))}
        updated = [unique[value] for value in raw]
        if updated == labels:
            break
        labels = updated
    return [(elements[index], len(adjacency[index]), labels[index]) for index in range(len(elements))]
