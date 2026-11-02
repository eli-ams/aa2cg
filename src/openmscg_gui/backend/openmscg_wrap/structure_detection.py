from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil

from .chemistry import molecular_mass
from .mapping import Bead, Molecule
from .trajectory_mapping import MASS_TO_ELEMENT, _prepare_atom, read_first_lammps_frame


@dataclass(frozen=True)
class DataAtom:
    atom_id: int
    molecule_id: int | None
    type_id: int
    charge: float
    element: str
    position: tuple[float, float, float]
    image: tuple[int, int, int]


@dataclass(frozen=True)
class DataBond:
    bond_id: int
    type_id: int
    first: int
    second: int


@dataclass
class LammpsStructure:
    atoms: dict[int, DataAtom]
    bonds: list[DataBond]
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass
class Component:
    molecule_id: int
    atom_ids: list[int]
    bonds: list[DataBond]
    mol: object | None = None
    smiles: str = ""
    formula: str = ""
    bond_order_status: str = ""


@dataclass
class SpeciesGroup:
    species_id: str
    name: str
    components: list[Component]
    smiles: str
    formula: str
    bond_order_status: str

    def summary(self) -> dict:
        representative = self.components[0]
        return {
            "id": self.species_id,
            "name": self.name,
            "formula": self.formula,
            "count": len(self.components),
            "smiles": self.smiles,
            "atom_count": len(representative.atom_ids),
            "heavy_atom_count": sum(
                1 for atom in representative.mol.GetAtoms() if atom.GetAtomicNum() != 1
            ),
            "representative_molecule_id": representative.molecule_id,
            "bond_order_status": self.bond_order_status,
        }


@dataclass
class StructureMapping:
    molecules: list[Molecule]
    records: list[tuple[int, str, int]]
    map_yaml: Path
    signature_path: Path
    species: list[dict]


SECTION_NAMES = {
    "Masses",
    "Atoms",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "Velocities",
    "Pair Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
}


def read_lammps_structure(path: str | Path) -> LammpsStructure:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LAMMPS structure file does not exist: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    sections, styles = _split_sections(lines)
    masses = _parse_masses(sections.get("Masses", []))
    atoms = _parse_atoms(sections.get("Atoms", []), styles.get("Atoms", ""), masses)
    if not atoms:
        raise ValueError(f"No atoms were found in the Atoms section of {path}.")
    bonds = _parse_bonds(sections.get("Bonds", []), atoms)
    expected_atoms = _header_count(lines, "atoms")
    expected_bonds = _header_count(lines, "bonds")
    if expected_atoms is not None and expected_atoms != len(atoms):
        raise ValueError(f"LAMMPS header declares {expected_atoms} atoms, parsed {len(atoms)}.")
    if expected_bonds is not None and expected_bonds != len(bonds):
        raise ValueError(f"LAMMPS header declares {expected_bonds} bonds, parsed {len(bonds)}.")
    ordered_ids = sorted(atoms)
    if ordered_ids != list(range(1, len(atoms) + 1)):
        raise ValueError(
            "Topology-driven mapping currently requires contiguous atom IDs starting at 1."
        )
    return LammpsStructure(atoms=atoms, bonds=bonds, box=_parse_box(lines))


def detect_species(path: str | Path) -> list[dict]:
    structure = read_lammps_structure(path)
    return [group.summary() for group in analyse_species(structure)]


def infer_structure_from_trajectory(
    trajectory_path: str | Path, output_path: str | Path
) -> dict:
    """Infer a LAMMPS molecular topology from the first dump frame with RDKit.

    Existing positive ``mol`` IDs limit inference to each declared molecule.
    Without them, RDKit perceives connectivity across the complete frame and
    connected components become molecule IDs. The generated file is deliberately
    retained so the inferred assumption is inspectable and reproducible.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    trajectory_path = Path(trajectory_path)
    output_path = Path(output_path)
    frame = read_first_lammps_frame(trajectory_path)
    columns = set(frame["columns"])
    if "id" not in columns:
        raise ValueError("Position-based topology inference requires an 'id' trajectory column.")
    if not ({"element", "el", "mass"} & columns):
        raise ValueError(
            "Position-based topology inference requires an element, el, or mass trajectory column."
        )

    atoms = [_prepare_atom(row, frame["bounds"], columns) for row in frame["rows"]]
    atoms.sort(key=lambda atom: atom["id"])
    unsupported = sorted({atom["element"] for atom in atoms} - set(MASS_TO_ELEMENT))
    if unsupported:
        raise ValueError(
            "Position-based topology inference does not yet have masses for elements: "
            + ", ".join(unsupported)
        )
    atom_ids = [atom["id"] for atom in atoms]
    if atom_ids != list(range(1, len(atoms) + 1)):
        raise ValueError(
            "Position-based topology inference requires contiguous atom IDs starting at 1."
        )
    row_by_id = {int(row["id"]): row for row in frame["rows"]}

    positive_molecule_ids = (
        "mol" in columns
        and all(int(row["mol"]) > 0 for row in frame["rows"])
    )
    if positive_molecule_ids:
        grouped: dict[int, list[dict]] = defaultdict(list)
        for atom in atoms:
            grouped[int(atom["mol"])].append(atom)
        inference_groups = [grouped[key] for key in sorted(grouped)]
        molecule_source = "trajectory mol IDs"
    else:
        inference_groups = [atoms]
        molecule_source = "inferred connected components"

    inferred_bonds: set[tuple[int, int]] = set()
    for group in inference_groups:
        editable = Chem.RWMol()
        for atom in group:
            editable.AddAtom(Chem.Atom(atom["element"]))
        mol = editable.GetMol()
        conformer = Chem.Conformer(len(group))
        for index, atom in enumerate(group):
            conformer.SetAtomPosition(index, atom["position"])
        mol.AddConformer(conformer)
        try:
            rdDetermineBonds.DetermineConnectivity(mol)
        except Exception as exc:
            raise ValueError(
                "RDKit could not infer covalent connectivity from the first trajectory frame: "
                f"{exc}"
            ) from exc
        for bond in mol.GetBonds():
            first = group[bond.GetBeginAtomIdx()]["id"]
            second = group[bond.GetEndAtomIdx()]["id"]
            inferred_bonds.add(tuple(sorted((first, second))))
        if positive_molecule_ids and len(group) > 1 and mol.GetNumBonds() == 0:
            raise ValueError(
                f"RDKit found no bonds for trajectory molecule {group[0]['mol']}. "
                "Check coordinates, image flags, and element/mass columns."
            )

    if positive_molecule_ids:
        molecule_by_atom = {atom["id"]: int(atom["mol"]) for atom in atoms}
    else:
        molecule_by_atom = _molecule_ids_from_inferred_bonds(atom_ids, inferred_bonds)

    type_keys = []
    for atom in atoms:
        row = row_by_id[atom["id"]]
        key = (row.get("type", atom["element"]), atom["element"])
        if key not in type_keys:
            type_keys.append(key)
    type_ids = {key: index for index, key in enumerate(type_keys, start=1)}

    lines = [
        "LAMMPS structure inferred by AA2CG from trajectory positions",
        "",
        f"{len(atoms)} atoms",
        f"{len(inferred_bonds)} bonds",
        "",
        f"{len(type_keys)} atom types",
        f"{1 if inferred_bonds else 0} bond types",
        "",
    ]
    for axis, (low, high) in zip("xyz", frame["bounds"]):
        lines.append(f"{low:.12g} {high:.12g} {axis}lo {axis}hi")
    lines.extend(["", "Masses", ""])
    for key, type_id in type_ids.items():
        element = key[1]
        lines.append(f"{type_id} {MASS_TO_ELEMENT[element]:.8g} # {element}")
    lines.extend(["", "Atoms # full", ""])
    for atom in atoms:
        row = row_by_id[atom["id"]]
        key = (row.get("type", atom["element"]), atom["element"])
        charge = float(row.get("q", row.get("charge", 0.0)))
        x, y, z = atom["position"]
        lines.append(
            f"{atom['id']} {molecule_by_atom[atom['id']]} {type_ids[key]} "
            f"{charge:.12g} {x:.12g} {y:.12g} {z:.12g} # {atom['element']}"
        )
    if inferred_bonds:
        lines.extend(["", "Bonds", ""])
        for bond_id, (first, second) in enumerate(sorted(inferred_bonds), start=1):
            lines.append(f"{bond_id} 1 {first} {second}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "atoms": len(atoms),
        "bonds": len(inferred_bonds),
        "molecules": len(set(molecule_by_atom.values())),
        "molecule_source": molecule_source,
        "output": str(output_path),
    }


def _molecule_ids_from_inferred_bonds(
    atom_ids: list[int], bonds: set[tuple[int, int]]
) -> dict[int, int]:
    adjacency = {atom_id: set() for atom_id in atom_ids}
    for first, second in bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)
    remaining = set(atom_ids)
    result = {}
    molecule_id = 0
    while remaining:
        molecule_id += 1
        stack = [min(remaining)]
        component = set()
        while stack:
            atom_id = stack.pop()
            if atom_id in component:
                continue
            component.add(atom_id)
            stack.extend(adjacency[atom_id] - component)
        remaining -= component
        result.update({atom_id: molecule_id for atom_id in component})
    return result


def _validate_atom_partition(mapped_atom_ids, expected_atom_ids) -> None:
    mapped_counts = Counter(mapped_atom_ids)
    expected_counts = Counter(expected_atom_ids)
    if mapped_counts == expected_counts:
        return
    expected_set = set(expected_counts)
    mapped_set = set(mapped_counts)
    missing = sorted(expected_set - mapped_set)
    unexpected = sorted(mapped_set - expected_set)
    duplicated = sorted(
        atom_id for atom_id, count in mapped_counts.items() if count > 1
    )
    raise ValueError(
        "United-atom mapping must assign every explicit atom exactly once. "
        f"Missing atom IDs: {missing[:10]}; unexpected atom IDs: {unexpected[:10]}; "
        f"duplicated atom IDs: {duplicated[:10]}."
    )


def analyse_species(structure: LammpsStructure) -> list[SpeciesGroup]:
    components = _components(structure)
    buckets: dict[tuple, list[Component]] = defaultdict(list)
    for component in components:
        buckets[_connectivity_key(component, structure)].append(component)

    groups: list[SpeciesGroup] = []
    for index, bucket in enumerate(
        sorted(buckets.values(), key=lambda values: min(values[0].atom_ids)), start=1
    ):
        representative = bucket[0]
        _perceive_chemistry(representative, structure)
        for component in bucket[1:]:
            component.smiles = representative.smiles
            component.formula = representative.formula
            component.bond_order_status = representative.bond_order_status
        groups.append(
            SpeciesGroup(
                species_id=f"species-{index}",
                name=f"species_{index}",
                components=bucket,
                smiles=representative.smiles,
                formula=representative.formula,
                bond_order_status=(
                    representative.bond_order_status
                ),
            )
        )
    return groups


def prepare_structure_mapping(
    structure_path: str | Path,
    output_dir: Path,
    expected_species: list[dict] | None = None,
    mapping_mode: str = "united_atom",
    custom_mapping_rules: list[dict] | None = None,
) -> StructureMapping:
    from rdkit import Chem

    from .dehydrogenize import _bead_signature, _ua_formula, ua_bead_type, write_signature

    output_dir.mkdir(parents=True, exist_ok=True)
    source_structure = Path(structure_path).resolve()
    structure = read_lammps_structure(source_structure)
    groups = analyse_species(structure)
    _apply_expected_names(groups, expected_species or [])

    if mapping_mode == "manual_custom":
        return _prepare_custom_structure_mapping(
            structure=structure,
            source_structure=source_structure,
            output_dir=output_dir,
            groups=groups,
            custom_mapping_rules=list(custom_mapping_rules or []),
        )
    if mapping_mode != "united_atom":
        raise ValueError(f"Unsupported mapping mode: {mapping_mode}")

    all_type_names = set()
    for group in groups:
        heavy_mol = Chem.RemoveHs(group.components[0].mol)
        for atom in heavy_mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                all_type_names.add(ua_bead_type(atom, heavy_mol))
    all_type_names = sorted(all_type_names)
    short_types = {value: f"A{index}" for index, value in enumerate(all_type_names, start=1)}
    signatures: dict[str, dict] = {}
    molecules: list[Molecule] = []
    sites: list[tuple[str, list[int], list[float], str]] = []

    for molecule_index, group in enumerate(groups, start=1):
        representative = group.components[0]
        explicit = representative.mol
        heavy = Chem.RemoveHs(explicit)
        explicit_heavy = [atom.GetIdx() for atom in explicit.GetAtoms() if atom.GetAtomicNum() != 1]
        if len(explicit_heavy) != heavy.GetNumAtoms():
            raise ValueError(f"Could not construct a united-atom representation for {group.name}.")

        template_beads: list[Bead] = []
        for local_index, (heavy_atom, explicit_index) in enumerate(
            zip(heavy.GetAtoms(), explicit_heavy), start=1
        ):
            attached_h = sorted(
                neighbor.GetIdx()
                for neighbor in explicit.GetAtomWithIdx(explicit_index).GetNeighbors()
                if neighbor.GetAtomicNum() == 1
            )
            member_indices = [explicit_index, *attached_h]
            formula = _ua_formula(heavy_atom.GetSymbol(), len(attached_h))
            canonical_type = ua_bead_type(heavy_atom, heavy)
            bead_type = short_types[canonical_type]
            signatures.setdefault(
                bead_type,
                _bead_signature(heavy_atom, heavy, formula, canonical_type),
            )
            template_beads.append(
                Bead(
                    name=f"{molecule_index}_{local_index}_{formula}_{bead_type}",
                    molecule_index=molecule_index,
                    local_index=local_index,
                    formula=formula,
                    bead_type=bead_type,
                    atom_ids=[representative.atom_ids[item] for item in member_indices],
                )
            )

        explicit_to_heavy = {explicit_id: index for index, explicit_id in enumerate(explicit_heavy)}
        connections = set()
        for bond in explicit.GetBonds():
            first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if first in explicit_to_heavy and second in explicit_to_heavy:
                connections.add(tuple(sorted((explicit_to_heavy[first], explicit_to_heavy[second]))))
        molecules.append(
            Molecule(
                name=group.name,
                number=len(group.components),
                formula=group.formula,
                beads_per_molecule=len(template_beads),
                beads=template_beads,
                bead_connections=sorted(connections),
            )
        )

        for component in group.components:
            alignment = _component_alignment(representative, component, structure)
            for bead in template_beads:
                representative_indices = [
                    representative.atom_ids.index(atom_id) for atom_id in bead.atom_ids
                ]
                target_atom_ids = [component.atom_ids[alignment[item]] for item in representative_indices]
                elements = [structure.atoms[atom_id].element for atom_id in target_atom_ids]
                site_name = f"site_{len(sites) + 1}"
                sites.append(
                    (
                        site_name,
                        [atom_id - 1 for atom_id in target_atom_ids],
                        [molecular_mass(element) for element in elements],
                        bead.bead_type,
                    )
                )

    mapped_atoms = [atom_index + 1 for _, indices, _, _ in sites for atom_index in indices]
    _validate_atom_partition(mapped_atoms, structure.atoms.keys())

    map_yaml = output_dir / "map.yaml"
    _write_explicit_map_yaml(map_yaml, sites, len(structure.atoms))
    bead_types = []
    for molecule in molecules:
        for bead in molecule.beads:
            if bead.bead_type not in bead_types:
                bead_types.append(bead.bead_type)
    records = [
        (index, bead_type, bead_types.index(bead_type))
        for index, (_, _, _, bead_type) in enumerate(sites, start=1)
    ]
    signature_path = output_dir / "bead_signature.json"
    write_signature(signature_path, signatures)
    species_path = output_dir / "detected_species.json"
    summaries = [group.summary() for group in groups]
    species_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    copied_structure = output_dir / "aa_structure.data"
    if source_structure != copied_structure.resolve():
        shutil.copy2(source_structure, copied_structure)
    return StructureMapping(molecules, records, map_yaml, signature_path, summaries)



def _prepare_custom_structure_mapping(
    structure: LammpsStructure,
    source_structure: Path,
    output_dir: Path,
    groups: list[SpeciesGroup],
    custom_mapping_rules: list[dict],
) -> StructureMapping:
    from rdkit import Chem

    from .dehydrogenize import write_signature

    if not custom_mapping_rules:
        raise ValueError("Manual custom mapping requires at least one saved bead rule.")

    rules_by_species: dict[str, list[dict]] = defaultdict(list)
    for rule in custom_mapping_rules:
        species_id = str(rule.get("species_id") or "")
        bead_label = str(rule.get("bead_label") or "")
        if not species_id or not bead_label:
            raise ValueError("Every custom bead rule must define species_id and bead_label.")
        if not rule.get("matches"):
            raise ValueError(f"Custom bead rule {bead_label} has no detected matches.")
        rules_by_species[species_id].append(rule)

    molecules: list[Molecule] = []
    sites: list[tuple[str, list[int], list[float], str]] = []
    signatures: dict[str, dict] = {}

    for molecule_index, group in enumerate(groups, start=1):
        rules = rules_by_species.get(group.species_id, [])
        if not rules:
            raise ValueError(
                f"Manual custom mapping has no bead rules for {group.name} ({group.species_id})."
            )
        representative = group.components[0]
        gui_to_rep = _custom_rule_atom_alignment(rules, representative)
        local_to_bead: dict[int, int] = {}
        template_beads: list[Bead] = []

        for rule in rules:
            bead_type = str(rule["bead_label"])
            _merge_custom_signature(signatures, bead_type, rule)
            for match in rule.get("matches", []):
                gui_atoms = [int(value) for value in match.get("atoms", [])]
                if not gui_atoms:
                    raise ValueError(f"Custom bead rule {bead_type} contains an empty match.")
                try:
                    representative_indices = sorted(gui_to_rep[index] for index in gui_atoms)
                except KeyError as exc:
                    raise ValueError(
                        f"Custom bead rule {bead_type} references atom index {exc.args[0]} outside the aligned species."
                    ) from exc
                for local_index in representative_indices:
                    if local_index in local_to_bead:
                        other = template_beads[local_to_bead[local_index]].bead_type
                        raise ValueError(
                            f"Custom bead rules overlap in {group.name}: representative atom {local_index} "
                            f"is assigned to both {other} and {bead_type}."
                        )
                    local_to_bead[local_index] = len(template_beads)
                atom_ids = [representative.atom_ids[index] for index in representative_indices]
                elements = [structure.atoms[atom_id].element for atom_id in atom_ids]
                formula = _formula_from_elements(elements)
                local_index = len(template_beads) + 1
                template_beads.append(
                    Bead(
                        name=f"{molecule_index}_{local_index}_{formula}_{bead_type}",
                        molecule_index=molecule_index,
                        local_index=local_index,
                        formula=formula,
                        bead_type=bead_type,
                        atom_ids=atom_ids,
                    )
                )

        expected = set(range(len(representative.atom_ids)))
        assigned = set(local_to_bead)
        if assigned != expected:
            missing = sorted(expected - assigned)
            extra = sorted(assigned - expected)
            raise ValueError(
                f"Manual custom mapping must assign every atom in {group.name} exactly once. "
                f"Missing representative atom indices: {missing}; unexpected indices: {extra}."
            )

        connections = set()
        atom_to_bead = {
            atom_id: bead_index
            for bead_index, bead in enumerate(template_beads)
            for atom_id in bead.atom_ids
        }
        for bond in representative.bonds:
            first = atom_to_bead[bond.first]
            second = atom_to_bead[bond.second]
            if first != second:
                connections.add(tuple(sorted((first, second))))

        molecules.append(
            Molecule(
                name=group.name,
                number=len(group.components),
                formula=group.formula,
                beads_per_molecule=len(template_beads),
                beads=template_beads,
                bead_connections=sorted(connections),
            )
        )

        for component in group.components:
            alignment = _component_alignment(representative, component, structure)
            for bead in template_beads:
                representative_indices = [representative.atom_ids.index(atom_id) for atom_id in bead.atom_ids]
                target_atom_ids = [component.atom_ids[alignment[index]] for index in representative_indices]
                elements = [structure.atoms[atom_id].element for atom_id in target_atom_ids]
                site_name = f"site_{len(sites) + 1}"
                sites.append(
                    (
                        site_name,
                        [atom_id - 1 for atom_id in target_atom_ids],
                        [molecular_mass(element) for element in elements],
                        bead.bead_type,
                    )
                )

    mapped_atoms = [atom_index + 1 for _, indices, _, _ in sites for atom_index in indices]
    _validate_atom_partition(mapped_atoms, structure.atoms.keys())

    map_yaml = output_dir / "map.yaml"
    _write_explicit_map_yaml(map_yaml, sites, len(structure.atoms))
    bead_types = []
    for molecule in molecules:
        for bead in molecule.beads:
            if bead.bead_type not in bead_types:
                bead_types.append(bead.bead_type)
    records = [
        (index, bead_type, bead_types.index(bead_type))
        for index, (_, _, _, bead_type) in enumerate(sites, start=1)
    ]
    signature_path = output_dir / "bead_signature.json"
    write_signature(signature_path, signatures)
    species_path = output_dir / "detected_species.json"
    summaries = [group.summary() for group in groups]
    species_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    copied_structure = output_dir / "aa_structure.data"
    if source_structure != copied_structure.resolve():
        shutil.copy2(source_structure, copied_structure)
    return StructureMapping(molecules, records, map_yaml, signature_path, summaries)


def _custom_rule_atom_alignment(rules: list[dict], representative: Component) -> dict[int, int]:
    from rdkit import Chem

    species_smiles = str(rules[0].get("species_smiles") or "")
    if not species_smiles:
        raise ValueError("Manual custom mapping rules must include species_smiles.")
    for rule in rules:
        if str(rule.get("species_smiles") or "") != species_smiles:
            raise ValueError("All custom rules for one species must have the same species_smiles.")
    query = Chem.MolFromSmiles(species_smiles)
    if query is None:
        raise ValueError(f"Could not parse custom-rule species SMILES: {species_smiles}")
    query = Chem.AddHs(query)
    target = representative.mol
    match = target.GetSubstructMatch(query)
    if len(match) != query.GetNumAtoms():
        raise ValueError(
            "Could not align custom-rule atom indices to the detected representative molecule. "
            "Run Stage 1 again and recreate the custom bead rules."
        )
    return {query_index: target_index for query_index, target_index in enumerate(match)}


def _merge_custom_signature(signatures: dict[str, dict], bead_type: str, rule: dict) -> None:
    comparable = {
        "descriptor": rule.get("descriptor", {}),
        "settings": rule.get("settings", {}),
        "signature": rule.get("signature", ""),
    }
    example = {
        "species_id": rule.get("species_id"),
        "species_name": rule.get("species_name"),
        "species_smiles": rule.get("species_smiles"),
    }
    if bead_type in signatures:
        current = signatures[bead_type]
        current_comparable = {
            "descriptor": current.get("descriptor", {}),
            "settings": current.get("settings", {}),
            "signature": current.get("signature", ""),
        }
        if current_comparable != comparable:
            raise ValueError(
                f"Custom bead label {bead_type} is used for different signatures. "
                "Use unique bead labels or make the descriptors identical."
            )
        examples = current.setdefault("examples", [])
        if example not in examples:
            examples.append(example)
        return
    signatures[bead_type] = {
        "source": "manual_custom",
        "bead_label": bead_type,
        **comparable,
        "examples": [example],
    }
def _split_sections(lines: list[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    sections: dict[str, list[str]] = defaultdict(list)
    styles: dict[str, str] = {}
    current: str | None = None
    for raw in lines:
        text = raw.strip()
        heading = text.split("#", 1)[0].strip()
        if heading in SECTION_NAMES:
            current = heading
            styles[heading] = text.split("#", 1)[1].strip().lower() if "#" in text else ""
            continue
        if current is not None:
            sections[current].append(raw)
    return dict(sections), styles


def _parse_masses(lines: list[str]) -> dict[int, tuple[float, str | None]]:
    result = {}
    for raw in lines:
        data, _, comment = raw.partition("#")
        values = data.split()
        if len(values) < 2:
            continue
        type_id, mass = int(values[0]), float(values[1])
        element = _element_from_comment(comment)
        result[type_id] = (mass, element)
    return result


def _parse_atoms(
    lines: list[str], style: str, masses: dict[int, tuple[float, str | None]]
) -> dict[int, DataAtom]:
    result = {}
    for raw in lines:
        data, _, comment = raw.partition("#")
        values = data.split()
        if not values:
            continue
        atom_id, molecule_id, type_id, charge, xyz, image = _decode_atom(values, style)
        element = _element_from_comment(comment)
        if element is None:
            if type_id not in masses:
                raise ValueError(
                    f"Atom type {type_id} has no Masses entry and no element comment."
                )
            mass, mass_element = masses[type_id]
            element = mass_element or _element_from_mass(mass)
        if atom_id in result:
            raise ValueError(f"Duplicate atom ID {atom_id} in LAMMPS data file.")
        result[atom_id] = DataAtom(
            atom_id, molecule_id, type_id, charge, element, xyz, image
        )
    return result


def _decode_atom(values: list[str], style: str):
    style = style.split()[0] if style else ""
    if style in {"full", "charge"} or (not style and len(values) in {7, 10}):
        atom_id = int(values[0])
        if style == "charge":
            molecule_id, type_id, charge, start = None, int(values[1]), float(values[2]), 3
        else:
            molecule_id, type_id, charge, start = int(values[1]), int(values[2]), float(values[3]), 4
    elif style in {"molecular", "bond", "angle"} or (not style and len(values) in {6, 9}):
        atom_id, molecule_id, type_id, charge, start = (
            int(values[0]), int(values[1]), int(values[2]), 0.0, 3
        )
    elif style == "atomic" or (not style and len(values) in {5, 8}):
        atom_id, molecule_id, type_id, charge, start = (
            int(values[0]), None, int(values[1]), 0.0, 2
        )
    else:
        raise ValueError(
            "Could not infer the LAMMPS atom style. Add an 'Atoms # full', "
            "'Atoms # molecular', or 'Atoms # atomic' section annotation."
        )
    xyz = tuple(float(value) for value in values[start : start + 3])
    if len(xyz) != 3:
        raise ValueError(f"Malformed atom row: {' '.join(values)}")
    image_values = values[start + 3 : start + 6]
    image = tuple(int(value) for value in image_values) if len(image_values) == 3 else (0, 0, 0)
    return atom_id, molecule_id, type_id, charge, xyz, image


def _parse_bonds(lines: list[str], atoms: dict[int, DataAtom]) -> list[DataBond]:
    result = []
    for raw in lines:
        values = raw.split("#", 1)[0].split()
        if not values:
            continue
        if len(values) < 4:
            raise ValueError(f"Malformed bond row: {raw}")
        bond = DataBond(*(int(value) for value in values[:4]))
        if bond.first not in atoms or bond.second not in atoms:
            raise ValueError(f"Bond {bond.bond_id} references a missing atom.")
        result.append(bond)
    return result


def _components(structure: LammpsStructure) -> list[Component]:
    adjacency: dict[int, set[int]] = {atom_id: set() for atom_id in structure.atoms}
    for bond in structure.bonds:
        adjacency[bond.first].add(bond.second)
        adjacency[bond.second].add(bond.first)

    molecule_ids = {atom.molecule_id for atom in structure.atoms.values()}
    use_molecule_ids = bool(molecule_ids) and all(
        molecule_id is not None and int(molecule_id) > 0 for molecule_id in molecule_ids
    )
    groups: list[tuple[int, list[int]]] = []
    if use_molecule_ids:
        by_molecule: dict[int, list[int]] = defaultdict(list)
        for atom in structure.atoms.values():
            by_molecule[int(atom.molecule_id)].append(atom.atom_id)
        groups = [(mol_id, sorted(atom_ids)) for mol_id, atom_ids in sorted(by_molecule.items())]
        atom_to_molecule = {
            atom_id: molecule_id for molecule_id, atom_ids in groups for atom_id in atom_ids
        }
        crossing = [
            bond for bond in structure.bonds
            if atom_to_molecule[bond.first] != atom_to_molecule[bond.second]
        ]
        if crossing:
            raise ValueError(
                f"Found {len(crossing)} bonds connecting different LAMMPS molecule IDs."
            )
    else:
        remaining = set(structure.atoms)
        component_id = 0
        while remaining:
            component_id += 1
            seed = min(remaining)
            stack, found = [seed], set()
            while stack:
                atom_id = stack.pop()
                if atom_id in found:
                    continue
                found.add(atom_id)
                stack.extend(adjacency[atom_id] - found)
            remaining -= found
            groups.append((component_id, sorted(found)))

    result = []
    for molecule_id, atom_ids in groups:
        atom_set = set(atom_ids)
        bonds = [
            bond for bond in structure.bonds
            if bond.first in atom_set and bond.second in atom_set
        ]
        result.append(Component(molecule_id, atom_ids, bonds))
    return result


def _connectivity_key(component: Component, structure: LammpsStructure) -> tuple:
    elements, adjacency = _component_graph(component, structure)
    labels: list[object] = [
        (
            elements[index],
            structure.atoms[atom_id].type_id,
            len(adjacency[index]),
        )
        for index, atom_id in enumerate(component.atom_ids)
    ]
    for _ in range(min(8, len(elements))):
        raw = [
            (labels[index], tuple(sorted((labels[neighbor] for neighbor in adjacency[index]), key=repr)))
            for index in range(len(elements))
        ]
        ranks = {value: rank for rank, value in enumerate(sorted(set(raw), key=repr))}
        updated = [ranks[value] for value in raw]
        if updated == labels:
            break
        labels = updated
    edge_types = Counter()
    local = {atom_id: index for index, atom_id in enumerate(component.atom_ids)}
    for bond in component.bonds:
        edge_types[
            tuple(sorted((repr(labels[local[bond.first]]), repr(labels[local[bond.second]]))))
            + (bond.type_id,)
        ] += 1
    total_charge = round(sum(structure.atoms[item].charge for item in component.atom_ids))
    return (
        tuple(sorted(Counter(elements).items())),
        tuple(sorted(Counter(map(repr, labels)).items())),
        tuple(sorted(edge_types.items(), key=repr)),
        total_charge,
    )


def _component_graph(component: Component, structure: LammpsStructure):
    local = {atom_id: index for index, atom_id in enumerate(component.atom_ids)}
    elements = [structure.atoms[atom_id].element for atom_id in component.atom_ids]
    adjacency = [set() for _ in component.atom_ids]
    for bond in component.bonds:
        first, second = local[bond.first], local[bond.second]
        adjacency[first].add(second)
        adjacency[second].add(first)
    return elements, adjacency


def _component_alignment(
    source: Component, target: Component, structure: LammpsStructure
) -> dict[int, int]:
    source_elements, source_adjacency = _component_graph(source, structure)
    target_elements, target_adjacency = _component_graph(target, structure)
    if len(source_elements) != len(target_elements):
        raise ValueError("Detected species instances have different atom counts.")

    def edge_types(component: Component):
        local = {atom_id: index for index, atom_id in enumerate(component.atom_ids)}
        return {
            tuple(sorted((local[bond.first], local[bond.second]))): bond.type_id
            for bond in component.bonds
        }

    source_edges = edge_types(source)
    target_edges = edge_types(target)

    def labels(component, elements, adjacency, edges):
        atom_types = [structure.atoms[atom_id].type_id for atom_id in component.atom_ids]
        values: list[object] = [
            (
                elements[index],
                atom_types[index],
                tuple(sorted(edges[tuple(sorted((index, neighbor)))] for neighbor in adjacency[index])),
            )
            for index in range(len(elements))
        ]
        for _ in range(min(8, len(elements))):
            raw = [
                (
                    values[index],
                    tuple(
                        sorted(
                            (
                                edges[tuple(sorted((index, neighbor)))],
                                repr(values[neighbor]),
                            )
                            for neighbor in adjacency[index]
                        )
                    ),
                )
                for index in range(len(elements))
            ]
            ranks = {value: rank for rank, value in enumerate(sorted(set(raw), key=repr))}
            updated = [ranks[value] for value in raw]
            if updated == values:
                break
            values = updated
        return values

    source_labels = labels(source, source_elements, source_adjacency, source_edges)
    target_labels = labels(target, target_elements, target_adjacency, target_edges)
    if Counter(map(repr, source_labels)) != Counter(map(repr, target_labels)):
        raise ValueError("Molecules grouped as one species have incompatible typed graphs.")
    candidates = {
        node: [
            other for other, label in enumerate(target_labels)
            if repr(label) == repr(source_labels[node])
        ]
        for node in range(len(source_labels))
    }
    order = sorted(
        range(len(source_labels)),
        key=lambda node: (len(candidates[node]), -len(source_adjacency[node]), node),
    )
    mapping: dict[int, int] = {}
    used = set()

    def search(position: int) -> bool:
        if position == len(order):
            return True
        node = order[position]
        for candidate in candidates[node]:
            if candidate in used:
                continue
            compatible = True
            for other, mapped in mapping.items():
                source_connected = other in source_adjacency[node]
                target_connected = mapped in target_adjacency[candidate]
                if source_connected != target_connected:
                    compatible = False
                    break
                if source_connected and (
                    source_edges[tuple(sorted((node, other)))]
                    != target_edges[tuple(sorted((candidate, mapped)))]
                ):
                    compatible = False
                    break
            if not compatible:
                continue
            mapping[node] = candidate
            used.add(candidate)
            if search(position + 1):
                return True
            used.remove(candidate)
            del mapping[node]
        return False

    if not search(0):
        raise ValueError("Could not align repeated instances of a detected molecular species.")
    return mapping


def _perceive_chemistry(component: Component, structure: LammpsStructure) -> None:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    editable = Chem.RWMol()
    for atom_id in component.atom_ids:
        editable.AddAtom(Chem.Atom(structure.atoms[atom_id].element))
    local = {atom_id: index for index, atom_id in enumerate(component.atom_ids)}
    for bond in component.bonds:
        editable.AddBond(local[bond.first], local[bond.second], Chem.BondType.SINGLE)
    mol = editable.GetMol()
    conformer = Chem.Conformer(len(component.atom_ids))
    for index, atom_id in enumerate(component.atom_ids):
        x, y, z = _unwrapped_position(structure.atoms[atom_id], structure.box)
        conformer.SetAtomPosition(index, (x, y, z))
    mol.AddConformer(conformer)
    charge = round(sum(structure.atoms[item].charge for item in component.atom_ids))
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
        Chem.SanitizeMol(mol)
        status = "inferred"
    except Exception:
        # Connectivity is still usable and authoritative. Sanitization of the
        # all-single graph lets simple saturated systems proceed while the UI
        # clearly marks uncertain chemistry.
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        status = "connectivity-only"
    heavy = Chem.RemoveHs(mol, sanitize=False)
    try:
        Chem.SanitizeMol(heavy)
    except Exception:
        pass
    component.mol = mol
    try:
        component.smiles = Chem.MolToSmiles(heavy, canonical=True)
    except Exception:
        component.smiles = ""
    component.formula = _formula_from_elements(
        [structure.atoms[atom_id].element for atom_id in component.atom_ids]
    )
    component.bond_order_status = status


def _write_explicit_map_yaml(path: Path, sites, atom_count: int) -> None:
    lines = ["site-types:"]
    for name, indices, weights, _ in sites:
        lines.extend(
            [
                f"  {name}:",
                f"    index: {indices}",
                f"    x-weight: {weights}",
                f"    f-weight: {[1.0 for _ in indices]}",
            ]
        )
    lines.extend(["system:", "  - anchor: 0", "    repeat: 1", f"    offset: {atom_count}", "    sites:"])
    lines.extend(f"      - [ {name}, 0 ]" for name, *_ in sites)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_expected_names(groups: list[SpeciesGroup], expected: list[dict]) -> None:
    lookup = {
        (item.get("formula"), int(item.get("count", 0)), item.get("smiles", "")): item
        for item in expected
    }
    for group in groups:
        key = (group.formula, len(group.components), group.smiles)
        if expected and key not in lookup:
            raise ValueError(
                "The selected structure.data no longer matches the species previously detected in the GUI. "
                "Run Detect species again."
            )
        if key in lookup:
            group.name = str(lookup[key].get("name") or group.name)


def _element_from_comment(comment: str) -> str | None:
    token = comment.strip().split()[0] if comment.strip() else ""
    match = re.match(r"^([A-Z][a-z]?)$", token)
    return match.group(1) if match and match.group(1) in MASS_TO_ELEMENT else None


def _element_from_mass(mass: float) -> str:
    element, reference = min(MASS_TO_ELEMENT.items(), key=lambda item: abs(item[1] - mass))
    if abs(reference - mass) > 0.75:
        raise ValueError(f"Could not infer an element from atom-type mass {mass}.")
    return element


def _formula_from_elements(elements: list[str]) -> str:
    counts = Counter(elements)
    if "C" in counts:
        order = ["C"]
        if "H" in counts:
            order.append("H")
        order.extend(sorted(element for element in counts if element not in {"C", "H"}))
    else:
        order = sorted(counts)
    return "".join(
        element + (str(counts[element]) if counts[element] != 1 else "")
        for element in order
    )


def _header_count(lines: list[str], noun: str) -> int | None:
    pattern = re.compile(rf"^\s*(\d+)\s+{re.escape(noun)}\s*$", re.IGNORECASE)
    for line in lines:
        match = pattern.match(line.split("#", 1)[0])
        if match:
            return int(match.group(1))
    return None


def _parse_box(lines: list[str]):
    values = {}
    for raw in lines:
        parts = raw.split("#", 1)[0].split()
        if len(parts) >= 4 and parts[2:4] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
            values[parts[2][0]] = (float(parts[0]), float(parts[1]))
    return tuple(values.get(axis, (0.0, 0.0)) for axis in "xyz")


def _unwrapped_position(atom: DataAtom, box):
    return tuple(
        coordinate + image * (high - low)
        for coordinate, image, (low, high) in zip(atom.position, atom.image, box)
    )
