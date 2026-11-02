"""Portable history/path safety. No optimizer or simulation needs to run."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from openmscg_gui.optimization_history import (StudyHistory, active_study, best_directory,
    contained, file_hash, read_json, save_json, setup_fingerprint)


class OptimizationHistoryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_new_studies_use_distinct_folders_and_keep_old_outputs(self):
        first = StudyHistory(self.root, 'fingerprint')
        first.activate()
        old = first.path / 'runs' / 'trial_0000'
        old.mkdir()
        (old / 'data').write_bytes(b'untouched')
        second = StudyHistory(self.root, 'fingerprint')
        second.activate()
        self.assertNotEqual(first.path, second.path)
        self.assertEqual((old / 'data').read_bytes(), b'untouched')
        self.assertEqual(active_study(self.root), second.path)

    def test_continuation_checks_compatibility_before_mutating_state(self):
        first = StudyHistory(self.root, 'old')
        first.activate()
        original = (first.path / 'state.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Cannot continue'):
            StudyHistory(self.root, 'changed', continuing=True)
        self.assertEqual((first.path / 'state.json').read_bytes(), original)
        restored = StudyHistory(self.root, 'old', continuing=True)
        self.assertEqual(restored.path, first.path)

    def test_interrupted_trial_numbers_are_not_reused(self):
        history = StudyHistory(self.root, 'same')
        history.reserve(0, 'explore')
        rows = list(history.records())
        self.assertEqual(rows[0]['state'], 'FAIL')
        self.assertTrue(rows[0]['user_attrs']['interrupted'])
        with self.assertRaises(FileExistsError):
            history.reserve(0, 'explore')
        history.reserve(1, 'refine')
        self.assertEqual(history.numbers(), [0, 1])

    def test_checkpoints_rebuild_full_optuna_history_through_public_api(self):
        history = StudyHistory(self.root, 'same')
        trial = SimpleNamespace(number=0, value=2., state=SimpleNamespace(name='COMPLETE'),
            params={'step': .2}, distributions={'step': dict(low=-1., high=1.)},
            user_attrs={'phase': 'explore'}, datetime_start=None, datetime_complete=None)
        optuna = SimpleNamespace(distributions=SimpleNamespace(distribution_to_json=json.dumps,
            json_to_distribution=json.loads), trial=SimpleNamespace(
                TrialState=SimpleNamespace(COMPLETE=trial.state, FAIL=SimpleNamespace(name='FAIL')),
                create_trial=lambda **values: SimpleNamespace(**values)))
        metadata = {'pair_corrections': [dict(table='Pair_A-A.table', amplitude=.2, center=4., sigma=1.)]}
        history.checkpoint(trial, metadata, {'score': 2.}, optuna)
        restored = []
        history.restore(SimpleNamespace(add_trial=restored.append), optuna)
        self.assertEqual(restored[0].params, trial.params)
        self.assertEqual(restored[0].distributions, trial.distributions)
        self.assertEqual(restored[0].value, 2.)
        with self.assertRaises(FileExistsError):
            history.checkpoint(trial, metadata, {'score': 2.}, optuna)

    def test_active_best_resolves_versioned_and_legacy_paths(self):
        self.assertEqual(best_directory(self.root), self.root / 'best')
        history = StudyHistory(self.root, 'same')
        destination = history.path / 'best' / 'trial_0007'
        destination.mkdir()
        history.state['best_path'] = destination.relative_to(self.root).as_posix()
        history.save()
        history.activate()
        self.assertEqual(best_directory(self.root), destination)
        with self.assertRaises(ValueError):
            contained(self.root, '../outside')

    def test_fingerprint_uses_contents_not_absolute_paths_or_output_logs(self):
        first, second = self.root / 'first', self.root / 'second'
        for folder in (first, second):
            folder.mkdir()
            (folder / 'input.data').write_text('same input')
            (folder / 'Pair_A-A.table').write_text('same table')
        (first / 'log.lammps').write_text('runtime log')
        self.assertEqual(setup_fingerprint(first, {'target': .81}), setup_fingerprint(second, {'target': .81}))
        original = setup_fingerprint(second, {'target': .81})
        (second / 'Pair_A-A.table').write_text('changed table')
        self.assertNotEqual(original, setup_fingerprint(second, {'target': .81}))
        self.assertNotEqual(setup_fingerprint(first, {'target': .81}), setup_fingerprint(first, {'target': .82}))


if __name__ == '__main__':
    unittest.main()
