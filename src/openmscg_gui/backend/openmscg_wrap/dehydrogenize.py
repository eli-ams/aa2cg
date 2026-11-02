from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.7-3.10
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

from .config import MoleculeConfig
from .trajectory_mapping import align_explicit_molecule_to_trajectory


def run_from_config(path: str | Path) -> None:
    if tomllib is None:
        raise RuntimeError(
            "TOML config files require Python 3.11+ or the optional 'tomli' package. "
            "For Python 3.7, use the top-level config.py + run.py workflow instead."
        )
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    run = raw.get("run", {})
    output_dir = Path(run.get("output_dir", "generated-mapping")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    system_title = run.get("title", config_path.stem)
    full_labels = bool(run.get("full_labels", False))
    molecules = raw.get("molecules", [])
    if not molecules:
        raise ValueError("Dehydrogenizer config needs at least one [[molecules]] item.")
    generate_mapping(system_title, molecules, output_dir, full_labels=full_labels)


def generate_mapping(
    system_title: str,
    molecules: list[dict],
    output_dir: Path,
    full_labels: bool = False,
    reference_trajectory: str | Path | None = None,
) -> Path:
    from rdkit import Chem

    atom_types = _collect_atom_types(molecules)
    type_names = {atom_type: f"A{index}" for index, atom_type in enumerate(sorted(atom_types), start=1)}

    atom_offset = 0
    ndx_blocks: list[str] = []
    signatures: dict[str, dict] = {}
    alignment_report: dict[str, dict] = {}
    for molecule_index, molecule in enumerate(molecules, start=1):
        parsed = Chem.MolFromSmiles(molecule["smiles"])
        if parsed is None:
            raise ValueError(f"Could not parse SMILES for {molecule['name']}.")
        mol = Chem.RemoveHs(parsed)
        mol_with_h = Chem.AddHs(mol)
        alignment = {index: index for index in range(mol_with_h.GetNumAtoms())}
        if reference_trajectory is not None:
            alignment = align_explicit_molecule_to_trajectory(
                mol_with_h,
                reference_trajectory,
                int(molecule["number"]),
            )
        alignment_report[molecule["name"]] = {
            "method": "trajectory_graph_isomorphism" if reference_trajectory is not None else "identity",
            "reference_trajectory": str(reference_trajectory) if reference_trajectory is not None else None,
            "rdkit_to_trajectory_atom": {
                str(rdkit_index + 1): trajectory_index + 1
                for rdkit_index, trajectory_index in sorted(alignment.items())
            },
        }
        new_order = [0] * mol_with_h.GetNumAtoms()
        for rdkit_index, trajectory_index in alignment.items():
            new_order[trajectory_index] = rdkit_index
        trajectory_ordered_mol = Chem.RenumberAtoms(mol_with_h, new_order)
        pdb_path = output_dir / f"{molecule['name']}.pdb"
        _write_pdb_from_mol(trajectory_ordered_mol, pdb_path)

        atom_to_bead_members = _atom_members_by_heavy_atom(mol, mol_with_h)
        heavy_atoms = sorted(mol.GetAtoms(), key=lambda atom: alignment[atom.GetIdx()])
        for bead_index, atom in enumerate(heavy_atoms, start=1):
            atom_type = ua_bead_type(atom, mol)
            member_atom_indices = [alignment[index] for index in atom_to_bead_members[atom.GetIdx()]]
            hydrogens = [
                index
                for index in member_atom_indices
                if trajectory_ordered_mol.GetAtomWithIdx(index).GetAtomicNum() == 1
            ]
            formula = _ua_formula(atom.GetSymbol(), len(hydrogens))
            label = atom_type if full_labels else type_names[atom_type]
            signatures.setdefault(label, _bead_signature(atom, mol, formula, atom_type))
            atom_ids = [index + 1 + atom_offset for index in member_atom_indices]
            ndx_blocks.append(
                f"[{molecule_index}_{bead_index}_{formula}_{label}]\n"
                + " ".join(str(idx) for idx in atom_ids)
                + "\n"
            )
        atom_offset += mol_with_h.GetNumAtoms()

    ndx_path = output_dir / f"{system_title}.ndx"
    ndx_path.write_text("\n".join(ndx_blocks) + "\n")
    write_signature(output_dir / "bead_signature.json", signatures)
    (output_dir / "atom_alignment.json").write_text(
        json.dumps(alignment_report, indent=2, sort_keys=True) + "\n"
    )
    return ndx_path


def write_signature(path: Path, signatures: dict[str, dict]) -> None:
    path.write_text(json.dumps(signatures, indent=2, sort_keys=True) + "\n")


def prepare_from_smiles(
    system_title: str,
    molecules: list[MoleculeConfig],
    output_dir: Path,
    full_labels: bool = False,
    reference_trajectory: str | Path | None = None,
) -> tuple[Path, Path, list[MoleculeConfig]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    molecule_dicts = []
    prepared_configs: list[MoleculeConfig] = []
    first_pdb: Path | None = None

    for index, molecule in enumerate(molecules, start=1):
        if not molecule.smiles:
            raise ValueError(f"Molecule {molecule.name!r} needs a SMILES string for automatic preparation.")
        molecule_dicts.append(
            {"name": molecule.name, "smiles": molecule.smiles, "number": molecule.number}
        )
        prepared_configs.append(molecule_config_from_smiles(molecule))
        pdb_path = output_dir / f"{molecule.name}.pdb"
        if index == 1:
            first_pdb = pdb_path

    if len(molecule_dicts) != 1:
        raise NotImplementedError("Automatic SMILES preparation currently supports one molecule type per run.")

    ndx_path = generate_mapping(
        system_title,
        molecule_dicts,
        output_dir,
        full_labels=full_labels,
        reference_trajectory=reference_trajectory,
    )
    if first_pdb is None:
        raise ValueError("No molecule PDB was generated.")
    return ndx_path, first_pdb, prepared_configs


def molecule_config_from_smiles(molecule: MoleculeConfig) -> MoleculeConfig:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    if not molecule.smiles:
        raise ValueError(f"Molecule {molecule.name!r} needs a SMILES string.")
    parsed = Chem.MolFromSmiles(molecule.smiles)
    if parsed is None:
        raise ValueError(f"Could not parse SMILES for {molecule.name!r}.")
    mol = Chem.RemoveHs(parsed)
    return MoleculeConfig(
        name=molecule.name,
        number=molecule.number,
        formula=rdMolDescriptors.CalcMolFormula(mol),
        beads_per_molecule=sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1),
        smiles=molecule.smiles,
    )


def _write_pdb_from_smiles(smiles: str, output_path: Path) -> None:
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    _write_pdb_from_mol(mol, output_path)


def _write_pdb_from_mol(mol, output_path: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    output_path.write_text(Chem.rdmolfiles.MolToPDBBlock(mol))


def _collect_atom_types(molecules: list[dict]) -> set[str]:
    from rdkit import Chem

    atom_types: set[str] = set()
    for molecule in molecules:
        parsed = Chem.MolFromSmiles(molecule["smiles"])
        if parsed is None:
            raise ValueError(f"Could not parse SMILES for {molecule['name']!r}.")
        mol = Chem.RemoveHs(parsed)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                atom_types.add(ua_bead_type(atom, mol))
    return atom_types


def ua_bead_type(atom, mol) -> str:
    symbol = atom.GetSymbol()
    hydrogen_count = atom.GetTotalNumHs()
    charge_suffix = _charge_suffix(atom.GetFormalCharge())
    hybridization = str(atom.GetHybridization())

    if symbol == "C":
        if atom.GetIsAromatic():
            base = "C_AR" if hydrogen_count == 0 else f"C_AR_H{hydrogen_count}"
            return base.replace("_H1", "_H") + charge_suffix
        if _has_double_bonded_neighbor(atom, mol, "O"):
            return f"C_CARBONYL_H{hydrogen_count}{charge_suffix}"
        return f"C_{hybridization}_H{hydrogen_count}{charge_suffix}"

    if symbol == "O":
        double_bonded_neighbors = _double_bonded_neighbors(atom, mol)
        if any(neighbor.GetSymbol() == "S" for neighbor in double_bonded_neighbors):
            return f"O_SULFOXIDE{charge_suffix}"
        if any(neighbor.GetSymbol() == "C" for neighbor in double_bonded_neighbors):
            return f"O_CARBONYL{charge_suffix}"
        if atom.GetIsAromatic():
            return f"O_AR_H{hydrogen_count}{charge_suffix}"
        return f"O_{hybridization}_H{hydrogen_count}{charge_suffix}"

    if symbol == "S":
        if atom.GetIsAromatic():
            return f"S_AR{charge_suffix}"
        double_bonded_oxygen_count = sum(
            neighbor.GetSymbol() == "O"
            for neighbor in _double_bonded_neighbors(atom, mol)
        )
        if double_bonded_oxygen_count == 1:
            return f"S_SULFOXIDE{charge_suffix}"
        if double_bonded_oxygen_count >= 2:
            return f"S_SULFONE{charge_suffix}"
        return f"S_{hybridization}_H{hydrogen_count}{charge_suffix}"

    if symbol == "N":
        if atom.GetIsAromatic():
            return f"N_AR_H{hydrogen_count}{charge_suffix}"
        return f"N_{hybridization}_H{hydrogen_count}{charge_suffix}"

    if symbol == "P":
        return f"P_{hybridization}_H{hydrogen_count}{charge_suffix}"

    return f"{symbol}_H{hydrogen_count}{charge_suffix}"


def _bead_signature(atom, mol, formula: str, canonical_label: str) -> dict:
    return {
        "canonical_label": canonical_label,
        "formula": formula,
        "element": atom.GetSymbol(),
        "atomic_number": int(atom.GetAtomicNum()),
        "aromatic": bool(atom.GetIsAromatic()),
        "hydrogens": int(atom.GetTotalNumHs()),
        "hybridization": str(atom.GetHybridization()),
        "formal_charge": int(atom.GetFormalCharge()),
        "degree": int(atom.GetDegree()),
        "ring": bool(atom.IsInRing()),
        "neighbor_elements": sorted(neighbor.GetSymbol() for neighbor in atom.GetNeighbors()),
        "ua_type": ua_bead_type(atom, mol),
    }


def _atom_members_by_heavy_atom(mol, mol_with_hydrogens) -> dict[int, list[int]]:
    members = {atom.GetIdx(): [atom.GetIdx()] for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1}
    for atom in mol_with_hydrogens.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        heavy_neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() != 1]
        if len(heavy_neighbors) != 1:
            raise ValueError(f"Hydrogen atom {atom.GetIdx()} does not have exactly one heavy neighbor.")
        members[heavy_neighbors[0].GetIdx()].append(atom.GetIdx())
    return members


def _ua_formula(symbol: str, hydrogen_count: int) -> str:
    if hydrogen_count == 0:
        return symbol
    if hydrogen_count == 1:
        return f"{symbol}H"
    return f"{symbol}H{hydrogen_count}"


def _charge_suffix(charge: int) -> str:
    if charge == 0:
        return ""
    return f"_Q{charge:+d}"


def _double_bonded_neighbors(atom, mol):
    from rdkit import Chem

    return [
        neighbor
        for neighbor in atom.GetNeighbors()
        if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
        == Chem.BondType.DOUBLE
    ]


def _has_double_bonded_neighbor(atom, mol, symbol: str) -> bool:
    return any(neighbor.GetSymbol() == symbol for neighbor in _double_bonded_neighbors(atom, mol))
