from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .lammps_generation import INPUT_FILE, SMOKE_TEST_MARKER, STRUCTURE_FILE


SMOKE_DIR = "smoke_test"
SUCCESS_FILE = "smoke-test.ok"


def prepare_smoke_test(system_root: str | Path) -> Path:
    """Copy the generated LAMMPS setup and replace production with ``run 0``."""
    root = Path(system_root)
    source = root / "output" / "lammps"
    destination = root / SMOKE_DIR
    for required in (source / INPUT_FILE, source / STRUCTURE_FILE):
        if not required.is_file():
            raise FileNotFoundError(
                f"Generated LAMMPS file is missing: {required}. Generate the setup in 6 · LAMMPS first."
            )

    if destination.is_symlink():
        raise RuntimeError(f"Refusing to replace symlinked smoke-test directory: {destination}")
    if destination.exists() and not destination.is_dir():
        raise RuntimeError(f"Smoke-test path exists but is not a directory: {destination}")
    if destination.is_dir():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)

    input_path = destination / INPUT_FILE
    lines = input_path.read_text(encoding="utf-8").splitlines()
    try:
        boundary = lines.index(SMOKE_TEST_MARKER)
    except ValueError as exc:
        raise ValueError(
            "The generated input.data has no smoke-test boundary. Regenerate the LAMMPS setup "
            "with the current OpenMSCG GUI backend."
        ) from exc
    smoke_lines = [
        *lines[:boundary],
        "",
        "# Zero-step initialization: evaluates the complete force-field setup without dynamics.",
        "run 0",
    ]
    input_path.write_text("\n".join(smoke_lines) + "\n", encoding="utf-8")
    return destination


def run_smoke_test(system_root: str | Path, command: list[str]) -> Path:
    if not command:
        raise ValueError("Configure a LAMMPS command before running the smoke test.")
    directory = prepare_smoke_test(system_root)
    print(f"Smoke-test directory: {directory}", flush=True)
    print(f"Command: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=directory, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"LAMMPS smoke test exited with code {result.returncode}.")
    success = directory / SUCCESS_FILE
    success.write_text(
        json.dumps(
            {
                "command": list(command),
                "input": INPUT_FILE,
                "mode": "run 0",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("LAMMPS accepted the generated structure, tables, and force-field setup.", flush=True)
    return directory
