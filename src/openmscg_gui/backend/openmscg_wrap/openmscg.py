from __future__ import annotations

from contextlib import contextmanager
from multiprocessing import Pool, Process
from pathlib import Path
import csv
import json
import os

from . import fitting
from .io import copy_streamed, ensure_dir


def count_lammpstrj_frames(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith("ITEM: TIMESTEP"):
                count += 1
    if count == 0:
        raise ValueError(f"No frames found in {path}.")
    return count


def convert_aa_to_cg(aa_trajectory: Path, map_yaml: Path, output_path: Path, work_dir: Path, cores: int) -> None:
    frame_count = count_lammpstrj_frames(aa_trajectory)
    workers = max(1, min(cores, frame_count))
    part_paths: list[Path] = []
    processes: list[Process] = []

    start = 0
    base_chunk, remainder = divmod(frame_count, workers)
    for index in range(1, workers + 1):
        frames = base_chunk + (1 if index <= remainder else 0)
        part_path = work_dir / f"cg_part_{index}.lammpstrj"
        part_paths.append(part_path)
        process = Process(
            target=_run_cgmap,
            args=(str(aa_trajectory), str(map_yaml), start, frames, str(part_path)),
        )
        processes.append(process)
        start += frames

    for process in processes:
        process.start()
    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"cgmap worker failed with exit code {process.exitcode}.")

    copy_streamed(part_paths, output_path, delete_sources=True)


def _run_cgmap(aa_trajectory: str, map_yaml: str, skip: int, frames: int, output_path: str) -> None:
    from mscg.cli import cgmap

    cgmap.main(
        map=map_yaml,
        traj=[f"{aa_trajectory},skip={skip},every=1,frames={frames}"],
        out=output_path,
    )


def run_histograms(
    topology: Path,
    trajectory: Path,
    histogram_dir: Path,
    histogram_config: dict,
    interactions_by_kind: dict[str, list[tuple[str, ...]]],
) -> None:
    from mscg.cli import cgib

    ensure_dir(histogram_dir)
    for kind, interactions in interactions_by_kind.items():
        for interaction in interactions:
            params = fitting.params_for(histogram_config, kind, interaction)
            spec = fitting.cgib_spec(params, interaction)
            kwargs = {
                "top": str(topology),
                "traj": str(trajectory),
                "cut": float(params["cutoff"]),
                kind.lower(): [spec],
                "save": "return",
            }
            hist = cgib.main(**kwargs)[0]
            name = f"{kind}-{'-'.join(interaction)}"
            _write_histogram_csv(histogram_dir / f"histogram_xy_{name}.csv", hist["data"])
            _write_histogram_plot(histogram_dir / f"histogram_xy_{name}.png", hist["data"], name)


def run_force_matching(
    topology: Path,
    trajectory: Path,
    model_dir: Path,
    fit_config: dict,
    interactions_by_kind: dict[str, list[tuple[str, ...]]],
) -> list[Path]:
    from mscg.cli import cgfm

    ensure_dir(model_dir)
    outputs: list[Path] = []
    for kind, interactions in interactions_by_kind.items():
        for interaction in interactions:
            params = fitting.params_for(fit_config, kind, interaction)
            spec = fitting.cgfm_spec(params, interaction)
            model_base = model_dir / f"result_{kind}_{'-'.join(interaction)}"
            kwargs = {
                "top": str(topology),
                "traj": str(trajectory),
                "cut": float(params["cutoff"]),
                kind.lower(): [spec],
                "save": str(model_base),
            }
            cgfm.main(**kwargs)
            outputs.append(model_base.with_suffix(".p"))
    return outputs


def dump_raw_tables(
    model_dir: Path,
    table_dir: Path,
    fit_config: dict,
    interactions_by_kind: dict[str, list[tuple[str, ...]]],
) -> list[Path]:
    from mscg.cli import cgdump

    ensure_dir(table_dir)
    written: list[Path] = []
    with pushd(table_dir):
        for kind, interactions in interactions_by_kind.items():
            for interaction in interactions:
                params = fitting.params_for(fit_config, kind, interaction)
                model_file = model_dir / f"result_{kind}_{'-'.join(interaction)}.p"
                spec = fitting.cgdump_spec(kind, params, interaction)
                cgdump.main(file=str(model_file), dump=[spec])
                table_name = f"{kind}_{'-'.join(interaction)}.table"
                written.append(table_dir / table_name)
    return written


def run_histogram_pipeline_parallel(
    topology: Path,
    trajectory: Path,
    histogram_dir: Path,
    fit_plan_path: Path,
    histogram_config: dict,
    fit_config: dict,
    auto_fit: dict,
    interactions_by_kind: dict[str, list[tuple[str, ...]]],
    cores: int,
) -> list[dict]:
    ensure_dir(histogram_dir)
    ensure_dir(fit_plan_path.parent)
    jobs = [
        (
            kind,
            interaction,
            str(topology),
            str(trajectory),
            str(histogram_dir),
            histogram_config,
            fit_config,
            auto_fit,
        )
        for kind, interactions in interactions_by_kind.items()
        for interaction in interactions
    ]
    if not jobs:
        raise ValueError("No histogram jobs were generated.")

    workers = max(1, min(int(cores), len(jobs)))
    if workers == 1:
        entries = [_run_histogram_job(job) for job in jobs]
    else:
        with Pool(processes=workers) as pool:
            entries = pool.map(_run_histogram_job, jobs)

    payload = {
        "version": 1,
        "topology": str(Path(topology)),
        "trajectory": str(Path(trajectory)),
        "entries": entries,
    }
    fit_plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entries


def run_force_matching_from_plan_parallel(
    topology: Path,
    trajectory: Path,
    model_dir: Path,
    table_dir: Path,
    fit_plan_path: Path,
    cores: int,
    method: str = "independent",
) -> list[Path]:
    ensure_dir(model_dir)
    ensure_dir(table_dir)
    if not fit_plan_path.is_file():
        raise FileNotFoundError(f"Missing fit plan from CG model stage: {fit_plan_path}")
    payload = json.loads(fit_plan_path.read_text(encoding="utf-8"))
    entries = list(payload.get("entries", []))
    if not entries:
        raise ValueError(f"Fit plan contains no interactions: {fit_plan_path}")

    if method == "coupled":
        return _run_force_plan_coupled(
            topology=topology,
            trajectory=trajectory,
            model_dir=model_dir,
            table_dir=table_dir,
            entries=entries,
        )
    if method != "independent":
        raise ValueError(f"Unsupported force-matching method: {method}")

    jobs = [
        (
            str(topology),
            str(trajectory),
            str(model_dir),
            str(table_dir),
            entry,
        )
        for entry in entries
    ]
    workers = max(1, min(int(cores), len(jobs)))
    if workers == 1:
        return [_run_force_plan_job(job) for job in jobs]
    with Pool(processes=workers) as pool:
        return pool.map(_run_force_plan_job, jobs)


def run_interaction_pipeline_parallel(
    topology: Path,
    trajectory: Path,
    histogram_dir: Path,
    model_dir: Path,
    table_dir: Path,
    histogram_config: dict,
    fit_config: dict,
    auto_fit: dict,
    interactions_by_kind: dict[str, list[tuple[str, ...]]],
    cores: int,
) -> list[Path]:
    fit_plan_path = Path(model_dir).parent / "work" / "fit_plan.json"
    run_histogram_pipeline_parallel(
        topology=topology,
        trajectory=trajectory,
        histogram_dir=histogram_dir,
        fit_plan_path=fit_plan_path,
        histogram_config=histogram_config,
        fit_config=fit_config,
        auto_fit=auto_fit,
        interactions_by_kind=interactions_by_kind,
        cores=cores,
    )
    return run_force_matching_from_plan_parallel(
        topology=topology,
        trajectory=trajectory,
        model_dir=model_dir,
        table_dir=table_dir,
        fit_plan_path=fit_plan_path,
        cores=cores,
    )


def _run_histogram_job(job) -> dict:
    from mscg.cli import cgib

    (
        kind,
        interaction,
        topology,
        trajectory,
        histogram_dir,
        histogram_config,
        fit_config,
        auto_fit,
    ) = job

    histogram_params = fitting.params_for(histogram_config, kind, interaction)
    histogram_spec = fitting.cgib_spec(histogram_params, interaction)
    histogram = cgib.main(
        top=topology,
        traj=trajectory,
        cut=float(histogram_params["cutoff"]),
        **{kind.lower(): [histogram_spec], "save": "return"},
    )[0]

    interaction_name = "-".join(interaction)
    histogram_name = f"{kind}-{interaction_name}"
    histogram_csv = Path(histogram_dir) / f"histogram_xy_{histogram_name}.csv"
    histogram_png = Path(histogram_dir) / f"histogram_xy_{histogram_name}.png"
    _write_histogram_csv(histogram_csv, histogram["data"])
    _write_histogram_plot(histogram_png, histogram["data"], histogram_name)

    fit_params = fitting.params_for(fit_config, kind, interaction)
    histogram_range = (
        None
        if kind == "Dihedral"
        else fitting.infer_auto_fit_range(histogram["data"], auto_fit)
    )
    if histogram_range is not None:
        fit_params = fitting.apply_auto_fit_range_to_params(
            fit_params,
            histogram_range,
            int(auto_fit.get("bins", 200)),
        )

    return {
        "name": f"{kind}_{interaction_name}",
        "histogram_name": histogram_name,
        "kind": kind,
        "interaction": list(interaction),
        "histogram_csv": str(histogram_csv),
        "histogram_png": str(histogram_png),
        "fit_params": fit_params,
        "auto_fit_range": histogram_range,
        "auto_fit_settings": dict(auto_fit),
    }


def _run_force_plan_job(job) -> Path:
    from mscg.cli import cgdump, cgfm

    topology, trajectory, model_dir, table_dir, entry = job
    kind = str(entry["kind"])
    interaction = tuple(str(item) for item in entry["interaction"])
    fit_params = dict(entry["fit_params"])
    interaction_name = "-".join(interaction)

    fit_spec = fitting.cgfm_spec(fit_params, interaction)
    model_base = Path(model_dir) / f"result_{kind}_{interaction_name}"
    cgfm.main(
        top=topology,
        traj=trajectory,
        cut=float(fit_params["cutoff"]),
        **{kind.lower(): [fit_spec], "save": str(model_base)},
    )

    table_path = Path(table_dir) / f"{kind}_{interaction_name}.table"
    with pushd(Path(table_dir)):
        cgdump.main(
            file=str(model_base.with_suffix(".p")),
            dump=[fitting.cgdump_spec(kind, fit_params, interaction)],
        )
    _write_table_xyy_plot(table_path, table_path.with_suffix(".png"))
    return table_path


def _run_force_plan_coupled(
    topology: Path,
    trajectory: Path,
    model_dir: Path,
    table_dir: Path,
    entries: list[dict],
) -> list[Path]:
    """Fit every planned interaction in one OpenMSCG normal equation."""
    from mscg.cli import cgdump, cgfm

    model_dir = Path(model_dir).resolve()
    table_dir = Path(table_dir).resolve()
    model_specs = {"pair": [], "bond": [], "angle": [], "dihedral": []}
    cutoffs = []
    for entry in entries:
        kind = str(entry["kind"])
        option = kind.lower()
        if option not in model_specs:
            raise ValueError(f"Unsupported coupled interaction kind: {kind}")
        interaction = tuple(str(item) for item in entry["interaction"])
        fit_params = dict(entry["fit_params"])
        model_specs[option].append(fitting.cgfm_spec(fit_params, interaction))
        cutoffs.append(float(fit_params["cutoff"]))

    if not cutoffs:
        raise ValueError("The coupled force-matching plan contains no models.")

    model_base = Path(model_dir) / "result_coupled"
    kwargs = {
        "top": str(topology),
        "traj": str(trajectory),
        "cut": max(cutoffs),
        "save": str(model_base),
    }
    kwargs.update({name: specs for name, specs in model_specs.items() if specs})
    print(
        "Coupled OpenMSCG regression: "
        + ", ".join(
            f"{name}={len(specs)}" for name, specs in model_specs.items() if specs
        )
        + f", cutoff={max(cutoffs):g}"
    )
    cgfm.main(**kwargs)

    checkpoint = model_base.with_suffix(".p")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"OpenMSCG did not write the coupled model checkpoint: {checkpoint}"
        )

    tables = []
    with pushd(Path(table_dir)):
        for entry in entries:
            kind = str(entry["kind"])
            interaction = tuple(str(item) for item in entry["interaction"])
            fit_params = dict(entry["fit_params"])
            table_path = Path(table_dir) / f"{kind}_{'-'.join(interaction)}.table"
            cgdump.main(
                file=str(checkpoint),
                dump=[fitting.cgdump_spec(kind, fit_params, interaction)],
            )
            if not table_path.is_file():
                raise FileNotFoundError(
                    f"OpenMSCG did not export the coupled table: {table_path}"
                )
            _write_table_xyy_plot(table_path, table_path.with_suffix(".png"))
            tables.append(table_path)
    return tables


def _run_interaction_job(job) -> Path:
    raise RuntimeError("Use _run_histogram_job and _run_force_plan_job for split interaction jobs.")


def _write_histogram_csv(path: Path, data) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "count"])
        for row in data:
            writer.writerow([row[0], row[1]])


def _write_histogram_plot(path: Path, data, title: str) -> None:
    from matplotlib import pyplot as plt

    x = [row[0] for row in data]
    y = [row[1] for row in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_table_xyy_plot(table_path: Path, plot_path: Path) -> None:
    from matplotlib import pyplot as plt

    x, energy, force = _read_table_xyy(table_path)
    fig, ax_energy = plt.subplots(figsize=(7, 4))
    ax_force = ax_energy.twinx()
    energy_line = ax_energy.plot(x, energy, color="tab:blue", label="energy")
    force_line = ax_force.plot(x, force, color="tab:red", label="force")
    ax_energy.set_xlabel("x")
    ax_energy.set_ylabel("energy")
    ax_force.set_ylabel("force")
    ax_energy.set_title(table_path.stem)
    lines = energy_line + force_line
    labels = [line.get_label() for line in lines]
    ax_energy.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def _read_table_xyy(table_path: Path) -> tuple[list[float], list[float], list[float]]:
    x: list[float] = []
    energy: list[float] = []
    force: list[float] = []
    with table_path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                float(parts[0])
                x.append(float(parts[1]))
                energy.append(float(parts[2]))
                force.append(float(parts[3]))
            except ValueError:
                continue
    if not x:
        raise ValueError(f"No table data found in {table_path}.")
    return x, energy, force


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
