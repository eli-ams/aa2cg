from __future__ import annotations

from pathlib import Path
import shutil


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def copy_streamed(paths: list[Path], destination: Path, delete_sources: bool = False) -> None:
    with destination.open("wb") as output:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output)
    if delete_sources:
        for path in paths:
            if path.exists():
                path.unlink()
