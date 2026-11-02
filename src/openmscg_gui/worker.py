from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import traceback

from .project import Project


def _workflow(project: Project, system_id: str):
    from .backend.openmscg_wrap.config import (
        config_from_minimal_inputs,
        config_from_structure_inputs,
    )

    system = project.system(system_id)
    if system.structure_file and system.detected_species:
        config = config_from_structure_inputs(
            structure_data=system.structure_file,
            aa_trajectory=system.trajectory,
            species=system.detected_species,
            title=system.name,
            output_dir=project.system_root(system) / "output",
            cores=system.effective_cores,
            mapping_mode=system.mapping_mode,
            custom_mapping_rules=system.custom_mapping_rules,
        )
    else:
        config = config_from_minimal_inputs(
            smiles=system.smiles,
            number=system.molecule_count,
            aa_trajectory=system.trajectory,
            title=system.name,
            output_dir=project.system_root(system) / "output",
            cores=system.effective_cores,
        )
    return replace(
        config,
        keep_work=True,
        clean_output=True,
        interaction_level=system.interaction_level,
    )


def run_cgmodel(project: Project, system_id: str) -> None:
    from .backend.openmscg_wrap.cli import build_cg_model

    system = project.system(system_id)
    structure = Path(system.structure_file)
    if not structure.is_file():
        raise FileNotFoundError(f"LAMMPS structure data does not exist: {structure}")
    if not system.detected_species:
        raise ValueError("Complete Stage 1 inspection before building the CG model.")
    trajectory = Path(system.trajectory)
    if not trajectory.is_file():
        raise FileNotFoundError(f"Trajectory does not exist: {trajectory}")
    if sum(int(item.get("count", 0)) for item in system.detected_species) < 1:
        raise ValueError("The detected species table contains no molecules.")

    root = project.system_root(system)
    root.mkdir(parents=True, exist_ok=True)
    workflow = _workflow(project, system_id)
    snapshot = {
        "system": asdict(system),
        "mapping": system.mapping_mode,
        "interaction_level": workflow.interaction_level,
        "output_dir": str(workflow.output_dir),
    }
    (root / "workflow-snapshot.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    build_cg_model(workflow)




def run_force_match(project: Project, system_id: str) -> None:
    from .backend.openmscg_wrap.cli import run_force_matching_workflow
    from .fit_settings import effective_fit_plan

    system = project.system(system_id)
    workflow = replace(_workflow(project, system_id), clean_output=False)
    output = project.system_root(system_id) / "output"
    required = [
        output / "work" / "top.in",
        output / "work" / "types_cg.lammpstrj",
        output / "work" / "fit_plan.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Complete Stage 2 CG model first. Missing: " + ", ".join(missing))
    method = system.force_match_method
    # Validate before the fitting workflow cleans any previous model/table files.
    plan = json.loads((output / "work" / "fit_plan.json").read_text(encoding="utf-8"))
    effective = effective_fit_plan(plan, system.fit_overrides, root=project.system_root(system))
    effective_path = output / "work" / "effective_fit_plan.json"
    effective_path.write_text(json.dumps(effective, indent=2) + "\n", encoding="utf-8")
    print(f"Force-matching method: {method}")
    for entry in effective.get('entries', []):
        if entry['name'] in system.fit_overrides:
            print(f"Custom fitting settings: {entry['name']} {system.fit_overrides[entry['name']]}")
    run_force_matching_workflow(workflow, method=method, fit_plan_path=effective_path)
    (output / "work" / "force_match.json").write_text(
        json.dumps({"method": method, "fit_overrides": system.fit_overrides,
                    "fit_plan": effective}, indent=2) + "\n",
        encoding="utf-8",
    )

def run_inspect(project: Project, system_id: str) -> None:
    from .backend.openmscg_wrap.structure_detection import (
        detect_species,
        infer_structure_from_trajectory,
    )
    from .backend.openmscg_wrap.trajectory_inspection import inspect_trajectory

    system = project.system(system_id)
    structure = Path(system.structure_file)
    if not structure.is_file():
        if not system.infer_bonds_from_trajectory:
            raise FileNotFoundError(f"LAMMPS structure data does not exist: {structure}")
        trajectory = Path(system.trajectory)
        if not trajectory.is_file():
            raise FileNotFoundError(f"Trajectory does not exist: {trajectory}")
        structure = project.system_root(system) / "input" / "inferred_structure.data"
        report = infer_structure_from_trajectory(trajectory, structure)
        system.structure_file = str(structure.resolve())
        system.structure_inferred = True
        system.infer_bonds_from_trajectory = False
        project.save()
        print(
            "RDKit inferred topology: "
            f"atoms={report['atoms']} bonds={report['bonds']} "
            f"molecules={report['molecules']} source={report['molecule_source']}"
        )
        print(f"Inferred structure saved to: {structure}")
    species = detect_species(structure)
    if not species:
        raise ValueError("No molecular species were detected.")
    previous_names = {
        (item.get("formula"), item.get("smiles"), int(item.get("count", 0))): item.get("name")
        for item in system.detected_species
    }
    for item in species:
        key = (item.get("formula"), item.get("smiles"), int(item.get("count", 0)))
        if previous_names.get(key):
            item["name"] = previous_names[key]
    system.detected_species = species
    system.molecule_count = sum(int(item["count"]) for item in species)
    system.smiles = species[0].get("smiles", "")
    system.trajectory_stats = inspect_trajectory(system.trajectory, structure)
    for stage in ("cgmodel", "force", "postprocess", "lammps", "optimize", "validate"):
        system.stages[stage] = "pending"
    project.merge_status = "pending"
    project.save()
    print(f"Detected {len(species)} species across {system.molecule_count} molecules:")
    for item in species:
        print(
            f"  {item['name']}: count={item['count']} formula={item['formula']} "
            f"smiles={item['smiles']} bond_orders={item['bond_order_status']}"
        )
    stats = system.trajectory_stats
    print(
        f"Trajectory: frames={stats['frames']} steps={stats['first_timestep']}.."
        f"{stats['last_timestep']} atoms={stats['atoms']} "
        f"box={stats['box_first']}"
    )
    for check in stats["checks"]:
        print(f"  [{check['status'].upper()}] {check['name']}: {check['detail']}")


def run_postprocess(project: Project, system_id: str) -> None:
    from .backend import table_processing

    output = project.system_root(system_id) / "output"
    table_processing.RAW_TABLE_DIR = output / "tables"
    table_processing.PROCESSED_TABLE_DIR = output / "processed_tables"
    table_processing.main(overrides=project.system(system_id).table_overrides)


def run_lammps(project: Project, system_id: str) -> None:
    from .backend.lammps_generation import generate_lammps

    workflow = replace(_workflow(project, system_id), clean_output=False)
    generate_lammps(workflow)


def run_smoke(project: Project, system_id: str) -> None:
    from .backend.lammps_smoke import run_smoke_test

    system_root = project.system_root(system_id)
    run_smoke_test(system_root, list(project.execution.lammps_command))


def run_optimize(project: Project, system_id: str, continuation=None) -> None:
    from .backend.corrector import config
    from .backend.corrector import run

    system_root = project.system_root(system_id)
    optimization = system_root / "optimization"
    config.OPTIMIZATION_DIR = optimization
    config.CONTINUE_RUN = continuation is not None
    config.ADD_EXPLORE, config.ADD_REFINE = continuation if continuation is not None else (0, 0)
    config.SOURCE_LAMMPS_DIR = system_root / "output" / "lammps"
    config.WORK_DIR = optimization / "runs"
    config.BEST_DIR = optimization / "best"
    config.N_TRIALS = int(project.execution.optimize_trials)
    config.LAMMPS_COMMAND = list(project.execution.lammps_command)
    config.PROPERTIES = deepcopy(project.system(system_id).optimization_properties)
    config.SIMULATION_PROTOCOL = deepcopy(project.system(system_id).optimization_protocol)
    config.TRIAL_STORAGE = deepcopy(project.system(system_id).trial_storage)
    config.SEARCH = deepcopy(project.system(system_id).optimization_search)
    if not config.PROPERTIES:
        raise ValueError("Select at least one optimization property in 7 · Refine (optional).")
    print("Optuna property objective:")
    for settings in config.PROPERTIES:
        target = settings.get("target")
        target_text = "first sampled value" if target is None else str(target)
        print(
            f"  {settings['column']}: target={target_text} "
            f"weight={settings.get('weight', 1.0)} "
            f"scales(tail={settings['tail_scale']}, drift={settings['drift_scale']}, "
            f"noise={settings['std_scale']}, slope={settings['slope_scale']}) "
            f"skip={settings.get('skip_fraction', 0.0)} "
            f"guards(min={settings.get('min')}, max={settings.get('max')}, "
            f"max_abs_drift={settings.get('max_abs_drift')}, "
            f"penalty={settings.get('hard_penalty', 1.0e6)})"
        )
    config.INCLUDE_AA_CG_IN_LOSS = bool(project.execution.include_structural_loss)
    config.AA_CG_TOPOLOGY = system_root / "output" / "work" / "top.in"
    config.AA_CG_REFERENCE_TRAJECTORY = (
        system_root / "output" / "work" / "types_cg.lammpstrj"
    )
    config.AA_CG_REFERENCE_DIR = optimization / "aa_cg_reference"
    run.main()


def run_validate(project: Project, system_id: str) -> None:
    from .backend.validation_metrics import validate_system

    system = project.system(system_id)
    cg_candidate = project.validation_trajectory(system)
    if cg_candidate is None:
        raise FileNotFoundError(
            "Validate needs a CG simulation trajectory. Run Refine, or run the generated "
            "LAMMPS setup externally to produce output/lammps/cg.lammpstrj. "
            "The smoke test does not generate a trajectory."
        )
    merged_candidate = None
    if project.merge_status == "complete" and system_id in project.merge_system_ids:
        candidate = project.root / "merge" / "best" / "systems" / system.name / "cg.lammpstrj"
        if candidate.is_file():
            merged_candidate = candidate
    aa_properties = None
    if system.aa_properties_file:
        candidate = Path(system.aa_properties_file).expanduser()
        aa_properties = candidate if candidate.is_absolute() else project.root / candidate
    validate_system(
        project.system_root(system),
        merged_candidate=merged_candidate,
        aa_properties_path=aa_properties,
        cg_candidate=cg_candidate,
        sampling_settings=system.validation_sampling,
    )


def run_merge(project: Project) -> None:
    from .backend.merger import config
    from .backend.merger import run

    selected = project.merge_system_ids or [system.id for system in project.systems]
    systems = []
    for system_id in selected:
        system = project.system(system_id)
        from .optimization_history import best_directory
        best = best_directory(project.system_root(system) / 'optimization')
        if not best.is_dir():
            raise FileNotFoundError(
                f"Optimized output missing for {system.name}: {best}"
            )
        systems.append({"name": system.name, "path": best, "weight": 1.0})
    if len(systems) < 2:
        raise ValueError("Select at least two optimized systems to merge.")

    merge_root = project.root / "merge"
    config.PROJECT_DIR = project.root
    config.INPUT_DIR = project.root / "systems"
    config.OUTPUT_DIR = merge_root / "combined_forcefield"
    config.RUNS_DIR = merge_root / "runs"
    config.BEST_DIR = merge_root / "best"
    config.SYSTEMS = systems
    config.RUN_OPTIMIZER = bool(project.execution.merge_optimize)
    config.N_TRIALS = int(project.execution.merge_trials)
    config.LAMMPS_COMMAND = list(project.execution.lammps_command)
    run.main()


STAGES = {
    "inspect": run_inspect,
    "detect": run_inspect,
    "cgmodel": run_cgmodel,
    "map": run_cgmodel,
    "force": run_force_match,
    "postprocess": run_postprocess,
    "lammps": run_lammps,
    "smoke": run_smoke,
    "optimize": run_optimize,
    "validate": run_validate,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenMSCG GUI background worker")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stage", choices=[*STAGES, "merge"], required=True)
    parser.add_argument("--system")
    parser.add_argument('--continue-study', action='store_true')
    parser.add_argument('--add-explore', type=int, default=0)
    parser.add_argument('--add-refine', type=int, default=20)
    args = parser.parse_args(argv)
    if args.continue_study and (args.stage != 'optimize' or args.add_explore < 0 or args.add_refine < 0
                               or args.add_explore + args.add_refine < 1):
        parser.error('Continue requires the optimize stage and at least one additional trial.')

    try:
        project = Project.load(args.project)
        print(f"Starting stage: {args.stage}", flush=True)
        if args.stage == "merge":
            run_merge(project)
        else:
            if not args.system:
                parser.error(f"--system is required for {args.stage}")
            if args.continue_study:
                run_optimize(project, args.system, (args.add_explore, args.add_refine))
            else:
                STAGES[args.stage](project, args.system)
        print(f"Stage completed: {args.stage}", flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())




