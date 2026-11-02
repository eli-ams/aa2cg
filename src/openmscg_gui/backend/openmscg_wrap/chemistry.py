from __future__ import annotations

import re


ATOMIC_MASSES = {
    "H": 1.00794,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "S": 32.065,
    "P": 30.973762,
    "F": 18.9984032,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.90447,
    "Si": 28.0855,
}


def molecular_formula_to_atoms(formula: str) -> list[str]:
    atoms: list[str] = []
    for symbol, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if symbol not in ATOMIC_MASSES:
            raise ValueError(f"Unsupported element in formula: {symbol}")
        atoms.extend([symbol] * int(count or 1))
    if not atoms:
        raise ValueError(f"Could not parse molecular formula: {formula}")
    return atoms


def molecular_mass(formula: str) -> float:
    return sum(ATOMIC_MASSES[atom] for atom in molecular_formula_to_atoms(formula))


def element_from_label(value: str) -> str:
    match = re.match(r"[A-Za-z]+", value)
    if not match:
        raise ValueError(f"No element-like prefix found in {value!r}")
    element = match.group()
    if len(element) > 1 and element not in ATOMIC_MASSES:
        element = element[0]
    return element
