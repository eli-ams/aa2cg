"""Validation settings and worker wiring; no scientific jobs are launched."""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openmscg_gui.project import Project, SystemSpec
from openmscg_gui.validation_sampling import DEFAULT_SAMPLING, normalize_sampling, sampled_indices


class ValidationSamplingTests(unittest.TestCase):
    def test_defaults_and_even_spacing(self):
        self.assertEqual(normalize_sampling(), dict(maximum_frames=150, discard_percent=0.0))
        indices = sampled_indices(1000)
        self.assertEqual(len(indices), 150)
        self.assertEqual((indices[0], indices[-1]), (0, 999))
        self.assertEqual(len(set(indices)), 150)
        gaps = [second - first for first, second in zip(indices, indices[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 1)
        self.assertEqual(sampled_indices(5), [0, 1, 2, 3, 4])
        self.assertEqual(sampled_indices(1, dict(discard_percent=95)), [0])
        self.assertEqual(sampled_indices(12, dict(maximum_frames=5, discard_percent=25)),
                         [3, 5, 7, 9, 11])
        edited = normalize_sampling()
        edited['maximum_frames'] = 200
        self.assertEqual(DEFAULT_SAMPLING['maximum_frames'], 150)

    def test_invalid_settings_rejected(self):
        for settings in (dict(maximum_frames=0), dict(maximum_frames=1), dict(maximum_frames=2.5),
                         dict(maximum_frames=1001), dict(maximum_frames=float('nan')),
                         dict(discard_percent=-1), dict(discard_percent=100),
                         dict(discard_percent=float('inf')), dict(discard_percent='bad'), dict(unknown=1)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                normalize_sampling(settings)
        with self.assertRaises(ValueError):
            sampled_indices(0)

    def test_per_system_persistence_and_legacy_defaults(self):
        self.assertEqual(SystemSpec.from_dict({}).validation_sampling, {})
        with TemporaryDirectory() as directory:
            project = Project.create(directory)
            first = project.add_system('First')
            second = project.add_system('Second')
            first.validation_sampling = dict(maximum_frames=200, discard_percent=15)
            project.save()
            loaded = Project.load(directory)
            self.assertEqual(loaded.system(first.id).validation_sampling, first.validation_sampling)
            self.assertEqual(loaded.system(second.id).validation_sampling, {})
            self.assertEqual(SystemSpec.from_dict(asdict(first)).validation_sampling, first.validation_sampling)

    def test_worker_forwards_saved_settings_without_launching_simulation(self):
        from openmscg_gui.worker import run_validate

        with TemporaryDirectory() as directory:
            project = Project(path=Path(directory) / 'project.json')
            system = project.add_system('Example')
            system.stages['lammps'] = 'complete'
            system.validation_sampling = dict(maximum_frames=200, discard_percent=10)
            candidate = project.system_root(system) / 'output' / 'lammps' / 'cg.lammpstrj'
            candidate.parent.mkdir(parents=True)
            candidate.touch()
            with patch('openmscg_gui.backend.validation_metrics.validate_system') as validate:
                run_validate(project, system.id)
            self.assertEqual(validate.call_args[1]['sampling_settings'], system.validation_sampling)
            self.assertEqual(validate.call_args[1]['cg_candidate'], candidate)


if __name__ == '__main__':
    unittest.main()
