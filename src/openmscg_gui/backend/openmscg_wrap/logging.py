from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import time


class RunLogger:
    def __init__(self, log_dir: Path, title: str):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title)
        self.path = log_dir / f"{stamp}_{safe_title}.log"

    def line(self, message: str) -> None:
        text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}"
        print(text)
        with self.path.open("a") as handle:
            handle.write(text + "\n")

    @contextmanager
    def activity(self, name: str):
        start = time.time()
        self.line(f"START {name}")
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.line(f"END   {name} ({elapsed:.2f}s)")


def log_config_summary(logger: RunLogger, config, interactions: dict | None = None, header: bool = True) -> None:
    if header:
        logger.line(f"CONFIG title={config.title}")
        logger.line(f"CONFIG output={config.output_dir}")
        logger.line(f"CONFIG cores={config.cores} clean={config.clean_output} keep_work={config.keep_work}")
        for molecule in config.molecules:
            mode = (
                "structure.data"
                if config.files.get("structure_data")
                else "smiles" if molecule.smiles else "pdb/ndx"
            )
            logger.line(f"CONFIG molecule={molecule.name} number={molecule.number} mode={mode}")
        logger.line(f"CONFIG aa_trajectory={config.files.get('aa_trajectory')}")
        if config.files.get("structure_data"):
            logger.line(f"CONFIG aa_structure={config.files['structure_data']}")
        logger.line(
            f"CONFIG auto_fit enabled={config.auto_fit.get('enabled', True)} "
            f"mode={config.auto_fit.get('mode', 'threshold')}"
        )
    if interactions:
        counts = " ".join(f"{kind}={len(items)}" for kind, items in interactions.items())
        logger.line(f"CONFIG interactions {counts}")
