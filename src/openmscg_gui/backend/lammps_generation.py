from __future__ import annotations

from pathlib import Path
import math
import shutil

from .openmscg_wrap.dehydrogenize import molecule_config_from_smiles
from .openmscg_wrap.mapping import (
    add_bead_connections, build_molecules, read_ndx, read_pdb, unique_bead_types,
)
from .openmscg_wrap.structure import (
    generate_structure_data,
    interactions_from_structure_data,
    masses_for_bead_types,
    read_section,
    validate_structure_bonds,
)


LAMMPS_DIR = Path("output/lammps")
PROCESSED_TABLE_DIR = Path("output/processed_tables/2_accepted")
STRUCTURE_FILE = "structure.data"
INPUT_FILE = "input.data"
SMOKE_TEST_MARKER = "# OPENMSCG_SMOKE_TEST_END"

TEMPERATURE = 298.15
PRESSURE = 1.0
TEMPERATURE_DAMPING = 100.0
PRESSURE_DAMPING = 1000.0
DUMP_ATTRIBUTES = "id type mol mass x y z ix iy iz vx vy vz fx fy fz c_kepatom"

EQUILIBRATION = {
    "timestep": 0.05,
    "steps": 50_000,
    "dump_every": 50,
    "thermo_every": 250,
}

PRODUCTION = {
    "timestep": 1.0,
    "steps": 500_000,
    "dump_every": 250,
    "thermo_every": 1000,
}

AVE_TIME = {
    "every": 1,
    "repeat": 99,
    "frequency": 100,
    "file": "simulation_data.out",
}


def generate_lammps(workflow) -> Path:
    """Generate a complete LAMMPS folder for a completed workflow."""
    output_dir = Path(workflow.output_dir)
    lammps_dir = output_dir / "lammps"
    lammps_dir.mkdir(parents=True, exist_ok=True)

    top_in = output_dir / "work" / "top.in"
    typed_cg = output_dir / "work" / "types_cg.lammpstrj"
    bead_signature = output_dir / "work" / "bead_signature.json"
    _require_file(top_in, "Missing top.in. Set KEEP_WORK = True and rerun the main workflow.")
    _require_file(typed_cg, "Missing types_cg.lammpstrj. Set KEEP_WORK = True and rerun the main workflow.")
    _require_file(bead_signature, "Missing bead_signature.json. Rerun the main workflow with the updated code.")

    molecules = _load_molecules(workflow)
    _validate_export_masses(molecules, output_dir / "work" / STRUCTURE_FILE)

    print(f"Writing LAMMPS setup to {lammps_dir}")
    structure_data = lammps_dir / STRUCTURE_FILE
    generate_structure_data(
        topology_file=top_in,
        trajectory_file=typed_cg,
        output_path=structure_data,
        title=workflow.title,
        molecules=molecules,
        interaction_level=workflow.interaction_level,
    )
    geometry = validate_structure_bonds(structure_data)
    print(
        f"Validated structure: atoms={geometry['atoms']} bonds={geometry['bonds']} "
        f"max_bond_length={geometry['max_bond_length']:.3f} A"
    )
    interactions = interactions_from_structure_data(structure_data)
    _copy_processed_tables(
        source_dir=output_dir / PROCESSED_TABLE_DIR.relative_to("output"),
        destination_dir=lammps_dir,
        expected_tables=_expected_table_names(interactions),
    )
    shutil.copy2(bead_signature, lammps_dir / "bead_signature.json")
    _write_lammps_input(
        output_path=lammps_dir / INPUT_FILE,
        structure_file=STRUCTURE_FILE,
        interactions=interactions,
    )
    print("Done.")
    print(f"Processed tables copied into: {lammps_dir}")
    print(f"Run with: lmp -in {INPUT_FILE}")
    return lammps_dir


def _load_molecules(workflow):
    if "structure_data" in workflow.files:
        from .openmscg_wrap.structure_detection import prepare_structure_mapping

        prepared_structure = workflow.output_dir / "prepared" / "aa_structure.data"
        source = (
            prepared_structure
            if prepared_structure.is_file()
            else _resolve_input_path(workflow.input_dir, workflow.files["structure_data"])
        )
        mapping = prepare_structure_mapping(
            source,
            workflow.output_dir / "prepared",
            expected_species=[
                {
                    "name": molecule.name,
                    "count": molecule.number,
                    "formula": molecule.formula,
                    "smiles": molecule.smiles or "",
                }
                for molecule in workflow.molecules
            ],
            mapping_mode=workflow.mapping_mode,
            custom_mapping_rules=workflow.custom_mapping_rules or [],
        )
        return mapping.molecules
    if any(molecule.smiles for molecule in workflow.molecules):
        ndx_path = workflow.output_dir / "prepared" / f"{workflow.title}.ndx"
        pdb_path = workflow.output_dir / "prepared" / f"{workflow.molecules[0].name}.pdb"
        molecule_configs = [_prepared_molecule_config(molecule) for molecule in workflow.molecules]
    else:
        ndx_path = _resolve_input_path(workflow.input_dir, workflow.files["ndx"])
        pdb_path = _resolve_input_path(workflow.input_dir, workflow.files["pdb"])
        molecule_configs = workflow.molecules

    _require_file(ndx_path, f"Missing mapping file: {ndx_path}")
    _require_file(pdb_path, f"Missing PDB file: {pdb_path}")

    beads = read_ndx(ndx_path)
    _, pdb_bonds = read_pdb(pdb_path)
    molecules = build_molecules(molecule_configs, beads)
    add_bead_connections(molecules, pdb_bonds)
    return molecules


def _validate_export_masses(molecules, mapped_structure):
    """Do not silently relabel or change masses after the CG model was built."""
    if not mapped_structure.is_file():
        return  # Older workflows may not retain this additional reference.
    rows = read_section(mapped_structure, "Masses")
    bead_types = unique_bead_types(molecules)
    masses = masses_for_bead_types(molecules, bead_types)
    expected = [(index, name) for index, name in enumerate(bead_types, start=1)]
    actual = [(int(row[0]), str(row[-1])) for row in rows]
    if (actual != expected or any(not math.isclose(float(row[1]), mass,
            rel_tol=1.e-7, abs_tol=1.e-5) for row, mass in zip(rows, masses))):
        raise ValueError("LAMMPS export bead types/masses do not match the built CG model. "
                         "Rebuild the CG model with the current mapping before exporting.")


def _prepared_molecule_config(molecule):
    if not molecule.smiles:
        return molecule
    return molecule_config_from_smiles(molecule)


def _write_lammps_input(output_path, structure_file, interactions):
    pair_names = interactions.get("Pair", [])
    bond_names = interactions.get("Bond", [])
    angle_names = interactions.get("Angle", [])
    dihedral_names = interactions.get("Dihedral", [])
    atom_style = _lammps_atom_style(interactions)
    lines = [
        "units real",
        f"atom_style {atom_style}",
        "neigh_modify delay 10 every 1",
        "",
        f"read_data {structure_file}",
        "",
        "pair_style table spline 1000",
    ]
    lines.extend(
        f"pair_coeff {i + 1} {j + 1} Pair_{a}-{b}.table {a}-{b}"
        for i, a in enumerate(_unique_pair_types(pair_names))
        for j, b in enumerate(_unique_pair_types(pair_names)[i:], start=i)
    )
    if bond_names:
        lines.extend(["", "bond_style table spline 1000"])
        lines.extend(
            f"bond_coeff {index} Bond_{a}-{b}.table {a}-{b}"
            for index, (a, b) in enumerate(bond_names, start=1)
        )
    if angle_names:
        lines.extend(["", "angle_style table spline 1000"])
        lines.extend(
            f"angle_coeff {index} Angle_{a}-{b}-{c}.table {a}-{b}-{c}"
            for index, (a, b, c) in enumerate(angle_names, start=1)
        )
    if dihedral_names:
        lines.extend(["", "dihedral_style table spline 1000"])
        lines.extend(
            f"dihedral_coeff {index} Dihedral_{a}-{b}-{c}-{d}.table {a}-{b}-{c}-{d}"
            for index, (a, b, c, d) in enumerate(dihedral_names, start=1)
        )

    lines.extend(_simulation_lines(atom_style))
    output_path.write_text("\n".join(lines) + "\n")


def _expected_table_names(interactions):
    names = []
    names.extend(f"Pair_{a}-{b}.table" for a, b in interactions.get("Pair", []))
    names.extend(f"Bond_{a}-{b}.table" for a, b in interactions.get("Bond", []))
    names.extend(f"Angle_{a}-{b}-{c}.table" for a, b, c in interactions.get("Angle", []))
    names.extend(
        f"Dihedral_{a}-{b}-{c}-{d}.table"
        for a, b, c, d in interactions.get("Dihedral", [])
    )
    return names


def _copy_processed_tables(source_dir, destination_dir, expected_tables):
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Processed table directory not found: {source_dir}. Run process_tables.py first."
        )
    missing = [name for name in expected_tables if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing processed accepted table files:\n"
            + "\n".join(str(source_dir / name) for name in missing)
        )
    for name in expected_tables:
        shutil.copy2(source_dir / name, destination_dir / name)


def _simulation_lines(atom_style):
    dump_attributes = DUMP_ATTRIBUTES
    if atom_style == "atomic":
        dump_attributes = " ".join(item for item in DUMP_ATTRIBUTES.split() if item != "mol")
    return [
        "",
        "variable R equal 0.00198722",
        "variable sysvol equal vol",
        "variable sysmass equal mass(all)/6.0221367e+023",
        "variable sysdensity equal v_sysmass/v_sysvol/1.0e-24",
        "variable time equal step*dt",
        "variable cella equal xhi-xlo",
        "variable cellb equal yhi-ylo",
        "variable cellc equal zhi-zlo",
        "variable etotal equal etotal",
        "variable pe equal pe",
        "variable ke equal ke",
        "variable evdwl equal evdwl",
        "variable coulomb equal ecoul",
        "variable sxx equal pxx",
        "variable syy equal pyy",
        "variable szz equal pzz",
        "variable syz equal pyz",
        "variable sxz equal pxz",
        "variable sxy equal pxy",
        "variable xhi equal xhi",
        "variable yhi equal yhi",
        "variable zhi equal zhi",
        "variable xlo equal xlo",
        "variable ylo equal ylo",
        "variable zlo equal zlo",
        "variable surfacetension equal 0.5*v_cellc*(v_szz-0.5*(v_sxx+v_syy))",
        "thermo_style custom step temp press vol v_sysdensity pe ke ebond eangle evdwl",
        "thermo_modify flush yes",
        f"thermo {PRODUCTION['thermo_every']}",
        "",
        SMOKE_TEST_MARKER,
        "",
        "compute kepatom all ke/atom",
        f"velocity all create {TEMPERATURE} 1234 rot no dist gaussian",
        (
            f"fix npt all npt temp {TEMPERATURE} {TEMPERATURE} {TEMPERATURE_DAMPING} "
            f"iso {PRESSURE} {PRESSURE} {PRESSURE_DAMPING}"
        ),
        f"timestep {PRODUCTION['timestep']}",
        (
            f"fix simulation_data all ave/time {AVE_TIME['every']} {AVE_TIME['repeat']} {AVE_TIME['frequency']} "
            "&\n"
            "v_time c_thermo_temp c_thermo_press v_sysvol v_sysdensity &\n"
            "v_cella v_cellb v_cellc v_etotal v_pe v_ke v_evdwl v_coulomb &\n"
            "v_sxx v_syy v_szz v_syz v_sxz v_sxy &\n"
            "v_xhi v_yhi v_zhi v_xlo v_ylo v_zlo v_surfacetension &\n"
            f"file {AVE_TIME['file']}"
        ),
        f"dump trajectories all custom {PRODUCTION['dump_every']} cg.lammpstrj {dump_attributes}",
        f"run {PRODUCTION['steps']} post no",
        "unfix simulation_data",
        "undump trajectories",
        "uncompute kepatom",
        "unfix npt",
    ]


def _unique_pair_types(pair_names):
    types = []
    for a, b in pair_names:
        if a not in types:
            types.append(a)
        if b not in types:
            types.append(b)
    return types


def _interaction_level(interactions):
    if interactions.get("Dihedral"):
        return "Dihedral"
    if interactions.get("Angle"):
        return "Angle"
    if interactions.get("Bond"):
        return "Bond"
    if interactions.get("Pair"):
        return "Pair"
    raise ValueError("No interactions found.")


def _lammps_atom_style(interactions):
    level = _interaction_level(interactions)
    if level == "Pair":
        return "atomic"
    if level == "Dihedral":
        return "molecular"
    return level.lower()


def _resolve_input_path(input_dir, value):
    path = Path(value)
    return path if path.is_absolute() else input_dir / path


def _require_file(path, message):
    if not Path(path).is_file():
        raise FileNotFoundError(message)
