from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.7-3.10
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


@dataclass(frozen=True)
class MoleculeConfig:
    name: str
    number: int
    formula: str | None = None
    beads_per_molecule: int | None = None
    smiles: str | None = None


@dataclass(frozen=True)
class WorkflowConfig:
    title: str
    input_dir: Path
    output_dir: Path
    keep_work: bool
    clean_output: bool
    interaction_level: str | None
    molecules: list[MoleculeConfig]
    files: dict[str, str]
    cores: int
    histograms: dict
    fitting: dict
    auto_fit: dict
    mapping_mode: str = "united_atom"
    custom_mapping_rules: list[dict] | None = None


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    if tomllib is None:
        raise RuntimeError(
            "TOML config files require Python 3.11+ or the optional 'tomli' package. "
            "For Python 3.7, use the top-level config.py + run.py workflow instead."
        )
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    run = raw.get("run", {})
    execution = raw.get("execution", {})
    input_dir = _resolve(Path.cwd(), run.get("input_dir", "."))
    output_dir = _resolve(Path.cwd(), run.get("output_dir", "output"))

    molecules = [
        MoleculeConfig(
            name=item["name"],
            number=int(item["number"]),
            formula=item.get("formula"),
            beads_per_molecule=int(item["beads_per_molecule"]) if "beads_per_molecule" in item else None,
            smiles=item.get("smiles"),
        )
        for item in raw.get("molecules", [])
    ]
    if not molecules:
        raise ValueError("Config must define at least one [[molecules]] entry.")

    return WorkflowConfig(
        title=run.get("title", config_path.stem),
        input_dir=input_dir,
        output_dir=output_dir,
        keep_work=bool(run.get("keep_work", False)),
        clean_output=bool(run.get("clean_output", False)),
        interaction_level=run.get("interaction_level"),
        molecules=molecules,
        files=raw.get("files", {}),
        cores=int(execution.get("cores", 1)),
        histograms=raw.get("histograms", {}),
        fitting=raw.get("fitting", {}),
        auto_fit=raw.get("auto_fit", {"enabled": True}),
        mapping_mode=str(raw.get("mapping", {}).get("mode", "united_atom")),
        custom_mapping_rules=list(raw.get("mapping", {}).get("custom_rules", [])),
    )


def config_from_minimal_inputs(
    smiles: str,
    number: int,
    aa_trajectory: str | Path,
    title: str = "system",
    output_dir: str | Path = "output",
    cores: int = 1,
) -> WorkflowConfig:
    return WorkflowConfig(
        title=title,
        input_dir=Path.cwd(),
        output_dir=Path(output_dir).resolve(),
        keep_work=False,
        clean_output=True,
        interaction_level="Angle",
        molecules=[MoleculeConfig(name=title, number=number, smiles=smiles)],
        files={"aa_trajectory": str(Path(aa_trajectory).resolve())},
        cores=cores,
        histograms=_default_histograms(),
        fitting=_default_fitting(),
        auto_fit={
            "enabled": True,
            "mode": "quantile",
            "histogram_threshold_percent": 5.0,
            "lower_quantile": 0.005,
            "upper_quantile": 0.995,
            "bins": 200,
        },
        mapping_mode="united_atom",
        custom_mapping_rules=[],
    )


def config_from_structure_inputs(
    structure_data: str | Path,
    aa_trajectory: str | Path,
    species: list[dict],
    title: str = "system",
    output_dir: str | Path = "output",
    cores: int = 1,
    mapping_mode: str = "united_atom",
    custom_mapping_rules: list[dict] | None = None,
) -> WorkflowConfig:
    if not species:
        raise ValueError("Detect at least one molecular species before creating the workflow.")
    return WorkflowConfig(
        title=title,
        input_dir=Path.cwd(),
        output_dir=Path(output_dir).resolve(),
        keep_work=False,
        clean_output=True,
        interaction_level="Angle",
        molecules=[
            MoleculeConfig(
                name=str(item.get("name") or f"species_{index}"),
                number=int(item["count"]),
                formula=item.get("formula"),
                beads_per_molecule=(
                    int(item["heavy_atom_count"])
                    if item.get("heavy_atom_count") is not None
                    else None
                ),
                smiles=item.get("smiles"),
            )
            for index, item in enumerate(species, start=1)
        ],
        files={
            "aa_trajectory": str(Path(aa_trajectory).resolve()),
            "structure_data": str(Path(structure_data).resolve()),
        },
        cores=cores,
        histograms=_default_histograms(),
        fitting=_default_fitting(),
        auto_fit={
            "enabled": True,
            "mode": "quantile",
            "histogram_threshold_percent": 5.0,
            "lower_quantile": 0.005,
            "upper_quantile": 0.995,
            "bins": 200,
        },
        mapping_mode=str(mapping_mode),
        custom_mapping_rules=list(custom_mapping_rules or []),
    )


def config_from_python_settings(settings) -> WorkflowConfig:
    molecules = [
        MoleculeConfig(
            name=item["name"],
            number=int(item["number"]),
            formula=item.get("formula"),
            beads_per_molecule=int(item["beads_per_molecule"]) if item.get("beads_per_molecule") is not None else None,
            smiles=item.get("smiles"),
        )
        for item in settings.MOLECULES
    ]
    return WorkflowConfig(
        title=settings.SYSTEM_TITLE,
        input_dir=Path.cwd(),
        output_dir=Path(settings.OUTPUT_DIR).resolve(),
        keep_work=bool(settings.KEEP_WORK),
        clean_output=bool(settings.CLEAN_OUTPUT),
        interaction_level=settings.INTERACTION_LEVEL,
        molecules=molecules,
        files={
            "aa_trajectory": str(Path(settings.AA_TRAJECTORY).resolve()),
            **({"ndx": settings.NDX_FILE} if getattr(settings, "NDX_FILE", None) else {}),
            **({"pdb": settings.PDB_FILE} if getattr(settings, "PDB_FILE", None) else {}),
        },
        cores=int(settings.CORES),
        histograms=settings.HISTOGRAMS,
        fitting=settings.FITTING,
        auto_fit=settings.AUTO_FIT,
        mapping_mode=getattr(settings, "MAPPING_MODE", "united_atom"),
        custom_mapping_rules=list(getattr(settings, "CUSTOM_MAPPING_RULES", [])),
    )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _default_histograms() -> dict:
    return {
        "Pair": {"all": {"min": 1.0, "max": 10.0, "cutoff": 15.0, "bins": 100}},
        "Bond": {"all": {"min": 1.0, "max": 2.0, "cutoff": 15.0, "bins": 100}},
        "Angle": {"all": {"min": 0.0, "max": 180.0, "cutoff": 15.0, "bins": 100}},
        "Dihedral": {"all": {"min": -180.0, "max": 180.0, "cutoff": 15.0, "bins": 180}},
    }


def _default_fitting() -> dict:
    return {
        "Pair": {"all": {"min": 2.4, "max": 10.0, "cutoff": 15.0, "model": "BSpline", "order": 6, "resolution": 0.05}},
        "Bond": {"all": {"min": 2.4, "max": 10.0, "cutoff": 15.0, "model": "BSpline", "order": 6, "resolution": 0.1}},
        "Angle": {"all": {"min": 0.0, "max": 180.0, "cutoff": 15.0, "model": "BSpline", "order": 6, "resolution": 0.1}},
        "Dihedral": {"all": {"min": -180.0, "max": 180.0, "cutoff": 15.0, "model": "BSpline", "order": 6, "resolution": 5.0}},
    }
