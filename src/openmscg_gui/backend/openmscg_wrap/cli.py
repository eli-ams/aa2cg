from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import WorkflowConfig, config_from_minimal_inputs, load_workflow_config
from . import dehydrogenize, openmscg
from .io import clean_dir, ensure_dir
from .logging import RunLogger, log_config_summary
from .mapping import (
    add_bead_connections,
    bead_type_records,
    build_molecules,
    read_ndx,
    read_pdb,
    replace_cg_type_ids,
    unique_bead_types,
    validate_bead_formulae,
    write_map_yaml,
)
from .structure import (
    generate_structure_data,
    interactions_from_structure_data,
    validate_structure_bonds,
)
from .structure_detection import prepare_structure_mapping
from .topology import write_topology
from .validation import (
    require_files,
    validate_pdb_against_config,
    validate_trajectory_atom_count,
    validate_trajectory_atom_ids,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openmscg-wrap")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run CG model building, histograms, force matching, and raw table export.")
    run_parser.add_argument("config", nargs="?", type=Path)
    run_parser.add_argument("--smiles")
    run_parser.add_argument("--number", type=int)
    run_parser.add_argument("--aa-trajectory", type=Path)
    run_parser.add_argument("--title", default="system")
    run_parser.add_argument("--output-dir", type=Path, default=Path("output"))
    run_parser.add_argument("--cores", type=int, default=1)

    prep_parser = subparsers.add_parser("dehydrogenize", help="Generate NDX/PDB mapping helpers from SMILES.")
    prep_parser.add_argument("config", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        if args.config:
            config = load_workflow_config(args.config)
        else:
            if not (args.smiles and args.number and args.aa_trajectory):
                parser.error("run needs either a config file or --smiles, --number, and --aa-trajectory.")
            config = config_from_minimal_inputs(
                smiles=args.smiles,
                number=args.number,
                aa_trajectory=args.aa_trajectory,
                title=args.title,
                output_dir=args.output_dir,
                cores=args.cores,
            )
        run_workflow(config)
    elif args.command == "dehydrogenize":
        dehydrogenize.run_from_config(args.config)
    return 0


def run_workflow(config: WorkflowConfig) -> None:
    if config.clean_output:
        clean_dir(config.output_dir)
    paths = _paths(config)
    logger = RunLogger(paths["logs"], config.title)
    for path in paths.values():
        ensure_dir(path) if path.suffix == "" else None

    aa_traj_path = _resolve_input_path(config.input_dir, config.files["aa_trajectory"])
    logger.line(f"Log file: {logger.path}")
    log_config_summary(logger, config)
    _build_cg_model(config, paths, logger, aa_traj_path)
    _force_match_from_cg_model(config, paths, logger)
    _cleanup_work(config, paths, logger)
    logger.line(f"DONE models={paths['models']}")
    logger.line(f"DONE tables={paths['tables']}")


def build_cg_model(config: WorkflowConfig) -> None:
    if config.clean_output:
        clean_dir(config.output_dir)
    paths = _paths(config)
    logger = RunLogger(paths["logs"], config.title)
    for path in paths.values():
        ensure_dir(path) if path.suffix == "" else None

    aa_traj_path = _resolve_input_path(config.input_dir, config.files["aa_trajectory"])
    logger.line(f"Log file: {logger.path}")
    log_config_summary(logger, config)
    _build_cg_model(config, paths, logger, aa_traj_path)
    logger.line(f"DONE prepared={paths['prepared']}")
    logger.line(f"DONE histograms={paths['histograms']}")
    logger.line(f"DONE fit_plan={paths['work'] / 'fit_plan.json'}")


def run_force_matching_workflow(
    config: WorkflowConfig,
    method: str = "independent",
    fit_plan_path: Path | None = None,
) -> None:
    paths = _paths(config)
    logger = RunLogger(paths["logs"], config.title)
    logger.line(f"Log file: {logger.path}")
    log_config_summary(logger, config)
    _force_match_from_cg_model(config, paths, logger, method=method, fit_plan_path=fit_plan_path)
    _cleanup_work(config, paths, logger)
    logger.line(f"DONE models={paths['models']}")
    logger.line(f"DONE tables={paths['tables']}")


def _build_cg_model(config: WorkflowConfig, paths: dict[str, Path], logger: RunLogger, aa_traj_path: Path) -> None:
    with logger.activity("prepare inputs"):
        signature_path = None
        structure_mapping = None
        if "structure_data" in config.files:
            structure_path = _resolve_input_path(config.input_dir, config.files["structure_data"])
            require_files([structure_path, aa_traj_path])
            structure_mapping = prepare_structure_mapping(
                structure_path,
                paths["prepared"],
                expected_species=[
                    {
                        "name": molecule.name,
                        "count": molecule.number,
                        "formula": molecule.formula,
                        "smiles": molecule.smiles or "",
                    }
                    for molecule in config.molecules
                ],
                mapping_mode=config.mapping_mode,
                custom_mapping_rules=config.custom_mapping_rules or [],
            )
            molecule_configs = config.molecules
            signature_path = structure_mapping.signature_path
        elif _needs_preparation(config):
            ndx_path, pdb_path, prepared_molecules = dehydrogenize.prepare_from_smiles(
                system_title=config.title,
                molecules=config.molecules,
                output_dir=paths["prepared"],
                reference_trajectory=aa_traj_path,
            )
            signature_path = paths["prepared"] / "bead_signature.json"
            molecule_configs = prepared_molecules
        else:
            ndx_path = _resolve_input_path(config.input_dir, config.files["ndx"])
            pdb_path = _resolve_input_path(config.input_dir, config.files["pdb"])
            molecule_configs = config.molecules

        if structure_mapping is None:
            require_files([ndx_path, pdb_path, aa_traj_path])

    with logger.activity("validate mapping"):
        if structure_mapping is not None:
            molecules = structure_mapping.molecules
            records = structure_mapping.records
        else:
            beads = read_ndx(ndx_path)
            atoms, pdb_bonds = read_pdb(pdb_path)
            molecules = build_molecules(molecule_configs, beads)
            validate_bead_formulae(molecules)
            validate_pdb_against_config(atoms, molecules)
            add_bead_connections(molecules, pdb_bonds)
            bead_types = unique_bead_types(molecules)
            records = bead_type_records(molecules, bead_types)
        validate_trajectory_atom_count(aa_traj_path, molecules)
        if structure_mapping is not None:
            validate_trajectory_atom_ids(
                aa_traj_path,
                sum(
                    molecule.number
                    * sum(len(bead.atom_ids) for bead in molecule.beads)
                    for molecule in molecules
                ),
            )

    with logger.activity("write map/topology"):
        map_yaml = paths["work"] / "map.yaml"
        topology = paths["work"] / "top.in"
        if structure_mapping is not None:
            shutil.copy2(structure_mapping.map_yaml, map_yaml)
        else:
            write_map_yaml(map_yaml, molecules, atoms)
        write_topology(topology, molecules, config.interaction_level)
        _write_or_copy_bead_signature(signature_path, paths["work"] / "bead_signature.json", molecules)

    with logger.activity("convert AA to CG"):
        raw_cg = paths["work"] / "cg_aa.lammpstrj"
        typed_cg = paths["work"] / "types_cg.lammpstrj"
        openmscg.convert_aa_to_cg(aa_traj_path, map_yaml, raw_cg, paths["work"], config.cores)
        replace_cg_type_ids(raw_cg, typed_cg, records)

    with logger.activity("write/parse structure.data"):
        structure_data = paths["work"] / "structure.data"
        generate_structure_data(
            topology_file=topology,
            trajectory_file=typed_cg,
            output_path=structure_data,
            title=config.title,
            molecules=molecules,
            interaction_level=config.interaction_level,
        )
        geometry = validate_structure_bonds(structure_data)
        logger.line(
            "STRUCTURE "
            f"atoms={geometry['atoms']} bonds={geometry['bonds']} "
            f"max_bond_length={geometry['max_bond_length']:.3f}A"
        )
        interactions = interactions_from_structure_data(structure_data)
        _write_interactions(paths["work"] / "interactions.json", interactions)
    log_config_summary(logger, config, interactions=interactions, header=False)

    with logger.activity(f"histogram jobs max_cores={config.cores}"):
        openmscg.run_histogram_pipeline_parallel(
            topology=topology,
            trajectory=typed_cg,
            histogram_dir=paths["histograms"],
            fit_plan_path=paths["work"] / "fit_plan.json",
            histogram_config=config.histograms,
            fit_config=config.fitting,
            auto_fit=config.auto_fit,
            interactions_by_kind=interactions,
            cores=config.cores,
        )


def _force_match_from_cg_model(
    config: WorkflowConfig,
    paths: dict[str, Path],
    logger: RunLogger,
    method: str = "independent",
    fit_plan_path: Path | None = None,
) -> None:
    topology = paths["work"] / "top.in"
    typed_cg = paths["work"] / "types_cg.lammpstrj"
    fit_plan = fit_plan_path if fit_plan_path is not None else paths["work"] / "fit_plan.json"
    require_files([topology, typed_cg, fit_plan])
    clean_dir(paths["models"])
    clean_dir(paths["tables"])
    with logger.activity(f"force matching method={method} max_cores={config.cores}"):
        openmscg.run_force_matching_from_plan_parallel(
            topology=topology,
            trajectory=typed_cg,
            model_dir=paths["models"],
            table_dir=paths["tables"],
            fit_plan_path=fit_plan,
            cores=config.cores,
            method=method,
        )


def _cleanup_work(config: WorkflowConfig, paths: dict[str, Path], logger: RunLogger) -> None:
    with logger.activity("cleanup"):
        if not config.keep_work:
            shutil.rmtree(paths["work"], ignore_errors=True)


def _paths(config: WorkflowConfig) -> dict[str, Path]:
    output = ensure_dir(config.output_dir)
    return {
        "output": output,
        "prepared": ensure_dir(output / "prepared"),
        "logs": ensure_dir(output / "logs"),
        "work": ensure_dir(output / "work"),
        "histograms": ensure_dir(output / "histograms"),
        "models": ensure_dir(output / "models"),
        "tables": ensure_dir(output / "tables"),
    }


def _needs_preparation(config: WorkflowConfig) -> bool:
    return any(molecule.smiles for molecule in config.molecules)


def _resolve_input_path(input_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return input_dir / path


def _write_interactions(path: Path, interactions: dict[str, list[tuple[str, ...]]]) -> None:
    payload = {kind: [list(item) for item in values] for kind, values in interactions.items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_or_copy_bead_signature(source: Path | None, destination: Path, molecules) -> None:
    if source is not None and source.is_file():
        shutil.copy2(source, destination)
        return
    signatures = {}
    for molecule in molecules:
        for bead in molecule.beads:
            signatures.setdefault(
                bead.bead_type,
                {
                    "canonical_label": bead.bead_type,
                    "formula": bead.formula,
                    "source": "fallback_from_ndx",
                },
            )
    destination.write_text(json.dumps(signatures, indent=2, sort_keys=True) + "\n")
