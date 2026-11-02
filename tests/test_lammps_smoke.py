from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

from openmscg_gui.backend.lammps_generation import SMOKE_TEST_MARKER
from openmscg_gui.backend.lammps_smoke import (
    SUCCESS_FILE,
    prepare_smoke_test,
    run_smoke_test,
)


class LammpsSmokeTests(unittest.TestCase):
    def _generated_setup(self, root: Path) -> Path:
        source = root / "output" / "lammps"
        source.mkdir(parents=True)
        (source / "structure.data").write_text("LAMMPS structure\n")
        (source / "Pair_A-A.table").write_text("table\n")
        (source / "input.data").write_text(
            "units real\n"
            "read_data structure.data\n"
            f"{SMOKE_TEST_MARKER}\n"
            "velocity all create 298.15 1234\n"
            "run 500000\n"
        )
        return source

    def test_prepare_copies_setup_and_replaces_production_with_run_zero(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._generated_setup(root)

            smoke = prepare_smoke_test(root)

            text = (smoke / "input.data").read_text()
            self.assertIn("run 0", text)
            self.assertNotIn(SMOKE_TEST_MARKER, text)
            self.assertNotIn("velocity all create", text)
            self.assertNotIn("run 500000", text)
            self.assertTrue((smoke / "structure.data").is_file())
            self.assertTrue((smoke / "Pair_A-A.table").is_file())
            self.assertIn("run 500000", (source / "input.data").read_text())

    def test_successful_command_writes_completion_marker(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._generated_setup(root)
            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "text=Path('input.data').read_text(); "
                    "assert 'run 0' in text and 'run 500000' not in text"
                ),
            ]

            smoke = run_smoke_test(root, command)

            self.assertTrue((smoke / SUCCESS_FILE).is_file())


if __name__ == "__main__":
    unittest.main()
