from __future__ import annotations

from pathlib import Path
import json

from . import config
from .table_io import canonical_table_name, normalize_labels, parse_table_name, table_files


class SystemInput:
    def __init__(
        self,
        name: str,
        root: Path,
        lammps_dir: Path,
        weight: float,
        label_map: dict[str, str],
        table_map: dict[str, str],
    ):
        self.name = name
        self.root = root
        self.lammps_dir = lammps_dir
        self.weight = weight
        self.label_map = label_map
        self.table_map = table_map


def discover_systems() -> list[SystemInput]:
    raw_systems = _configured_or_discovered_systems()
    systems = []

    for item in raw_systems:
        name = item["name"]
        root = Path(item["path"])
        weight = float(item.get("weight", 1.0))
        lammps_dir = _find_lammps_dir(root)
        label_map = _label_map_for_system(name, lammps_dir)
        table_map = _build_table_map(lammps_dir, label_map)
        systems.append(SystemInput(name, root, lammps_dir, weight, label_map, table_map))

    if not systems:
        raise FileNotFoundError(f"No systems found in {config.INPUT_DIR}")
    return systems


def _configured_or_discovered_systems() -> list[dict[str, object]]:
    if config.SYSTEMS:
        return [
            {
                "name": str(item["name"]),
                "path": _resolve_user_path(Path(item["path"])),
                "weight": float(item.get("weight", 1.0)),
            }
            for item in config.SYSTEMS
        ]

    if not config.INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {config.INPUT_DIR}")
    return [
        {"name": path.name, "path": path, "weight": 1.0}
        for path in sorted(config.INPUT_DIR.iterdir())
        if path.is_dir()
    ]


def _resolve_user_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return config.PROJECT_DIR / path


def _find_lammps_dir(root: Path) -> Path:
    for candidate in config.LAMMPS_SOURCE_CANDIDATES:
        folder = root / candidate
        if _is_lammps_folder(folder):
            return folder
    raise FileNotFoundError(
        f"No LAMMPS folder found in {root}. Expected input.data, structure.data, and .table files."
    )


def _is_lammps_folder(folder: Path) -> bool:
    return (
        folder.is_dir()
        and (folder / "input.data").is_file()
        and (folder / "structure.data").is_file()
        and bool(list(folder.glob("*.table")))
    )


def _label_map_for_system(
    system_name: str,
    lammps_dir: Path,
) -> dict[str, str]:
    signature_path = lammps_dir / config.BEAD_SIGNATURE_FILE
    if not signature_path.is_file():
        raise FileNotFoundError(
            f"Missing required bead signature for {system_name}: {signature_path}. "
            "Rerun GUI stages 1 through 4 for this system."
        )

    signatures = json.loads(signature_path.read_text())
    if not isinstance(signatures, dict):
        raise ValueError(f"Bead signature file must contain an object: {signature_path}")

    label_map: dict[str, str] = {}
    for local_label, signature in signatures.items():
        if not isinstance(signature, dict):
            raise ValueError(f"Invalid signature for {local_label!r} in {signature_path}")
        canonical = signature.get("canonical_label")
        if not canonical:
            raise ValueError(f"Missing canonical_label for {local_label!r} in {signature_path}")
        label_map[str(local_label)] = str(canonical)
    return label_map


def _build_table_map(lammps_dir: Path, label_map: dict[str, str]) -> dict[str, str]:
    result = {}
    for path in table_files(lammps_dir):
        kind, labels = parse_table_name(path)
        missing = [label for label in labels if label not in label_map]
        if missing:
            raise ValueError(
                f"Table {path.name} uses labels missing from {config.BEAD_SIGNATURE_FILE}: "
                + ", ".join(missing)
            )
        mapped = tuple(label_map[label] for label in labels)
        canonical_labels = normalize_labels(kind, mapped)
        result[path.name] = canonical_table_name(kind, canonical_labels)
    return result
