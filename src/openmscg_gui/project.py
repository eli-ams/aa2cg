from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import uuid

from .optimization import default_optimization_properties
from .resources import available_cores


PROJECT_FILENAME = "openmscg-project.json"


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return value or "system"


@dataclass
class SystemSpec:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "molecule"
    smiles: str = ""
    molecule_count: int = 1
    trajectory: str = ""
    structure_file: str = ""
    aa_properties_file: str = ""
    structure_inferred: bool = False
    infer_bonds_from_trajectory: bool = False
    detected_species: list[dict] = field(default_factory=list)
    trajectory_stats: dict = field(default_factory=dict)
    cores: int = field(default_factory=available_cores)
    cores_auto: bool = True
    interaction_level: str = "Angle"
    force_match_method: str = "independent"
    fit_overrides: dict[str, dict] = field(default_factory=dict)
    table_overrides: dict[str, dict] = field(default_factory=dict)
    invalidated_stages: list[str] = field(default_factory=list)
    mapping_mode: str = "manual_custom"
    custom_mapping_rules: list[dict] = field(default_factory=list)
    optimization_properties: list[dict] = field(default_factory=default_optimization_properties)
    optimization_protocol: dict = field(default_factory=dict)
    trial_storage: dict = field(default_factory=dict)
    optimization_search: dict = field(default_factory=dict)
    validation_sampling: dict = field(default_factory=dict)
    stages: dict[str, str] = field(
        default_factory=lambda: {
            "inspect": "pending",
            "cgmodel": "pending",
            "force": "pending",
            "postprocess": "pending",
            "lammps": "pending",
            "smoke": "pending",
            "optimize": "pending",
            "validate": "pending",
        }
    )

    @classmethod
    def from_dict(cls, raw: dict) -> "SystemSpec":
        known = {key: raw[key] for key in cls.__dataclass_fields__ if key in raw}
        # Projects created before coupled force matching existed retain the
        # independent method that generated their current artifacts.
        known.setdefault("force_match_method", "independent")
        # Migrate the old single-worker default; retain larger explicit limits.
        # New projects can explicitly save manual one-core execution too.
        known.setdefault("cores_auto", int(raw.get("cores", 1)) <= 1)
        system = cls(**known)
        if system.force_match_method not in {"coupled", "independent"}:
            system.force_match_method = "independent"
        defaults = cls().stages
        raw_stages = dict(raw.get("stages", {}))
        if "map" in raw_stages:
            raw_stages.setdefault("cgmodel", raw_stages["map"])
            raw_stages.setdefault("force", raw_stages["map"])
            raw_stages.pop("map", None)
        system.stages = {**defaults, **raw_stages}
        if "inspect" not in raw.get("stages", {}) and system.detected_species:
            system.stages["inspect"] = "complete"
        return system

    @property
    def effective_cores(self) -> int:
        available = available_cores()
        return available if self.cores_auto else max(1, min(int(self.cores), available))

    @property
    def directory_name(self) -> str:
        return f"system-{self.id[:8]}"


@dataclass
class ExecutionSettings:
    lammps_command: list[str] = field(
        default_factory=lambda: ["mpiexec", "-np", "16", "lmp", "-in", "input.data"]
    )
    optimize_trials: int = 30
    include_structural_loss: bool = False
    merge_optimize: bool = True
    merge_trials: int = 30

    @classmethod
    def from_dict(cls, raw: dict) -> "ExecutionSettings":
        known = {key: raw[key] for key in cls.__dataclass_fields__ if key in raw}
        return cls(**known)


@dataclass
class Project:
    name: str = "OpenMSCG project"
    systems: list[SystemSpec] = field(default_factory=list)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    merge_system_ids: list[str] = field(default_factory=list)
    merge_status: str = "pending"
    merge_invalidated: bool = False
    version: int = 2
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    path: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def create(cls, directory: str | Path, name: str | None = None) -> "Project":
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        project = cls(name=name or directory.name)
        project.path = directory / PROJECT_FILENAME
        project.save()
        return project

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path).resolve()
        if path.is_dir():
            path = path / PROJECT_FILENAME
        raw = json.loads(path.read_text(encoding="utf-8"))
        project = cls(
            name=raw.get("name", path.parent.name),
            systems=[SystemSpec.from_dict(item) for item in raw.get("systems", [])],
            execution=ExecutionSettings.from_dict(raw.get("execution", {})),
            merge_system_ids=list(raw.get("merge_system_ids", [])),
            merge_status=raw.get("merge_status", "pending"),
            merge_invalidated=bool(raw.get("merge_invalidated", False)),
            version=int(raw.get("version", 1)),
            created=raw.get("created", datetime.now().isoformat(timespec="seconds")),
        )
        project.path = path
        for system in project.systems:
            for key in ('trajectory', 'structure_file', 'aa_properties_file'):
                value = getattr(system, key)
                if value and not (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()):
                    setattr(system, key, str((project.root / Path(value).expanduser()).resolve()))
        return project

    @property
    def root(self) -> Path:
        if self.path is None:
            raise RuntimeError("Project has not been saved yet.")
        return self.path.parent

    def system(self, system_id: str) -> SystemSpec:
        for system in self.systems:
            if system.id == system_id:
                return system
        raise KeyError(f"Unknown system: {system_id}")

    def system_root(self, system: SystemSpec | str) -> Path:
        item = self.system(system) if isinstance(system, str) else system
        return self.root / "systems" / item.directory_name

    def validation_trajectory(self, system: SystemSpec) -> Path | None:
        """Choose a current CG result; refinement is not required for a baseline run."""
        if system.stages.get("lammps") != "complete":
            return None
        root = self.system_root(system)
        candidates = []
        if self.merge_status == "complete" and system.id in self.merge_system_ids:
            candidates.append(self.root / "merge" / "best" / "systems" / system.name / "cg.lammpstrj")
        if system.stages.get("optimize") == "complete":
            from .optimization_history import best_directory
            candidates.append(best_directory(root / 'optimization') / 'cg.lammpstrj')
        candidates.append(root / "output" / "lammps" / "cg.lammpstrj")
        return next((path for path in candidates if path.is_file()), None)

    def add_system(self, name: str = "molecule") -> SystemSpec:
        system = SystemSpec(name=name)
        self.systems.append(system)
        self.save()
        return system

    def remove_system(self, system_id: str) -> None:
        self.systems = [item for item in self.systems if item.id != system_id]
        self.merge_system_ids = [item for item in self.merge_system_ids if item != system_id]
        self.save()

    def save(self) -> None:
        if self.path is None:
            raise RuntimeError("Project path is not set.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload.pop("path", None)
        # Keep project-owned inputs relocatable on disk, while callers continue
        # receiving absolute paths in memory. External inputs stay explicit.
        for system in payload['systems']:
            for key in ('trajectory', 'structure_file', 'aa_properties_file'):
                value = system.get(key)
                if value and Path(value).is_absolute():
                    try:
                        system[key] = Path(value).resolve().relative_to(self.root.resolve()).as_posix()
                    except ValueError:
                        pass
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def split_command(text: str) -> list[str]:
    import shlex

    parts = shlex.split(text, posix=False)
    return [
        part[1:-1] if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'" else part
        for part in parts
    ]


def display_command(parts: list[str]) -> str:
    return " ".join(parts)


