"""CPU defaults and optional refinement; no scientific jobs are launched."""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from openmscg_gui.project import Project, SystemSpec
from openmscg_gui.resources import available_cores


class ResourcesTests(unittest.TestCase):
    def test_windows_cpu_count_and_linux_affinity(self):
        for platform, expected in [
            (SimpleNamespace(cpu_count=lambda: 16), 16),
            (SimpleNamespace(cpu_count=lambda: 32, sched_getaffinity=lambda pid: set(range(16))), 16),
            (SimpleNamespace(cpu_count=lambda: 32, process_cpu_count=lambda: 8), 8),
            (SimpleNamespace(cpu_count=lambda: None), 1),
            (SimpleNamespace(cpu_count=lambda: 16,
                             sched_getaffinity=Mock(side_effect=OSError)), 16),
        ]:
            with self.subTest(expected=expected), patch('openmscg_gui.resources.os', platform):
                self.assertEqual(available_cores(), expected)

    def test_new_and_legacy_single_core_projects_use_auto(self):
        with patch('openmscg_gui.project.available_cores', return_value=16):
            for system in (SystemSpec(), SystemSpec.from_dict({}),
                           SystemSpec.from_dict({'cores': 1})):
                self.assertTrue(system.cores_auto)
                self.assertEqual(system.effective_cores, 16)

    def test_manual_limits_and_explicit_one_core_survive_round_trip(self):
        with patch('openmscg_gui.project.available_cores', return_value=16):
            legacy = SystemSpec.from_dict({'cores': 4})
            self.assertFalse(legacy.cores_auto)
            self.assertEqual(legacy.effective_cores, 4)
            manual = SystemSpec(cores=1, cores_auto=False)
            restored = SystemSpec.from_dict(asdict(manual))
            self.assertFalse(restored.cores_auto)
            self.assertEqual(restored.effective_cores, 1)
            self.assertEqual(SystemSpec(cores=64, cores_auto=False).effective_cores, 16)

    def test_auto_resolves_on_current_machine(self):
        system = SystemSpec.from_dict({'cores': 32, 'cores_auto': True})
        with patch('openmscg_gui.project.available_cores', return_value=16):
            self.assertEqual(system.effective_cores, 16)
        with patch('openmscg_gui.project.available_cores', return_value=8):
            self.assertEqual(system.effective_cores, 8)


class OptionalRefinementTests(unittest.TestCase):
    def test_baseline_allowed_without_refine_and_stale_outputs_skipped(self):
        with TemporaryDirectory() as directory:
            project = Project(path=Path(directory) / 'project.json')
            system = SystemSpec()
            project.systems.append(system)
            system.stages['lammps'] = 'complete'
            root = project.system_root(system)
            baseline = root / 'output' / 'lammps' / 'cg.lammpstrj'
            best = root / 'optimization' / 'best' / 'cg.lammpstrj'
            merged = project.root / 'merge' / 'best' / 'systems' / system.name / 'cg.lammpstrj'
            self.assertIsNone(project.validation_trajectory(system))  # Smoke test alone is insufficient.
            for path in (baseline, best, merged):
                path.parent.mkdir(parents=True)
                path.touch()
            self.assertEqual(project.validation_trajectory(system), baseline)
            system.stages['optimize'] = 'complete'
            self.assertEqual(project.validation_trajectory(system), best)
            project.merge_status = 'complete'
            project.merge_system_ids = [system.id]
            self.assertEqual(project.validation_trajectory(system), merged)
            project.merge_status = 'pending'
            system.stages['optimize'] = 'pending'
            self.assertEqual(project.validation_trajectory(system), baseline)
            system.stages['lammps'] = 'pending'
            self.assertIsNone(project.validation_trajectory(system))


if __name__ == '__main__':
    unittest.main()
