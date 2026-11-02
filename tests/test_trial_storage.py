"""Trial-retention regression cases. No LAMMPS/Optuna process is required."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openmscg_gui.project import SystemSpec
from openmscg_gui.trial_storage import TrialStorage, normalize_trial_storage


class TrialStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root / 'runs'
        self.runs.mkdir()

    def finish(self, storage, number, score, failed=False, state='COMPLETE', extra=None):
        directory = self.runs / 'trial_{:04d}'.format(number)
        directory.mkdir()
        for name in ('cg.lammpstrj', 'structure.data', 'simulation_data.out', 'Pair_A-A.table',
                     'input.data', 'correction.json', 'optuna_stdout.log', 'log.lammps',
                     'bead_signature.json'):
            (directory / name).write_text('test data', encoding='utf-8')
        plots = directory / 'table_plots'
        plots.mkdir()
        (plots / 'pair.png').write_bytes(b'image')
        data = dict(score=score, failed=failed, properties=[{'mean_value': 1.}])
        data.update(extra or {})
        (directory / 'score.json').write_text(json.dumps(data), encoding='utf-8')
        trial = SimpleNamespace(number=number, value=score, state=SimpleNamespace(name=state),
            params={'amplitude': number * .1}, user_attrs={}, datetime_start=None, datetime_complete=None)
        storage(None, trial)
        return directory

    def full_trials(self):
        return {int(p.name.split('_')[1]) for p in self.runs.iterdir()
                if p.is_dir() and (p / 'cg.lammpstrj').exists()}

    def test_default_keeps_all_and_records_every_trial(self):
        storage = TrialStorage(self.runs)
        for number, score in enumerate([100., 95., 110.]):
            self.finish(storage, number, score)
        self.assertEqual(self.full_trials(), {0, 1, 2})
        self.assertEqual(storage.best_number, 1)
        self.assertEqual(json.loads((self.runs / 'trial_0002' / 'trial_record.json').read_text())
                         ['parameters'], {'amplitude': .2})

    def test_milestones_accumulate_improvement_and_always_keep_best(self):
        storage = TrialStorage(self.runs, {'mode': 'milestones', 'threshold_percent': 5.})
        for number, score in enumerate([100., 98., 96., 95., 94., 110., 90.]):
            self.finish(storage, number, score)
        self.assertEqual(storage.milestones, {0, 3, 6})
        self.assertEqual(self.full_trials(), {0, 3, 6})
        self.finish(storage, 7, 89.)  # Less than 5%, still preserve the current best.
        self.assertEqual(self.full_trials(), {0, 3, 6, 7})
        self.finish(storage, 8, 89.)  # Ties do not replace the best.
        self.assertEqual(self.full_trials(), {0, 3, 6, 7})

    def test_best_only_preserves_lightweight_records_and_failed_logs(self):
        storage = TrialStorage(self.runs, {'mode': 'best'})
        first = self.finish(storage, 0, 100.)
        self.finish(storage, 1, 90.)
        failed = self.finish(storage, 2, 1.e12, failed=True)
        self.assertEqual(self.full_trials(), {1})
        for directory in (first, failed):
            self.assertFalse(directory.exists())
            number = int(directory.name.split('_')[1])
            archive = storage.archives[number]
            for name in ('input.data', 'correction.json', 'score.json', 'trial_record.json',
                         'optuna_stdout.log', 'log.lammps', 'bead_signature.json', 'storage.json'):
                self.assertTrue((archive / name).exists(), name)
            for name in ('simulation_data.out', 'structure.data', 'table_plots', 'Pair_A-A.table'):
                self.assertFalse((archive / name).exists(), name)
        self.assertFalse(json.loads((storage.archives[2] / 'trial_record.json').read_text())['successful'])

    def test_no_success_keeps_diagnostics_and_failed_penalties_are_not_candidates(self):
        storage = TrialStorage(self.runs, {'mode': 'best'})
        self.finish(storage, 0, 1.e12, failed=True)
        self.finish(storage, 1, 0., state='FAIL')
        self.finish(storage, 2, float('nan'))
        self.finish(storage, 3, 1.e12, extra={'properties': [{'mean_value': float('nan')}]})
        self.assertEqual(self.full_trials(), set())
        self.assertFalse(any(self.runs.iterdir()))
        self.assertEqual(set(storage.archives), {0, 1, 2, 3})
        self.assertIsNone(storage.best_number)
        self.finish(storage, 4, 2.e12)  # A valid score can exceed the failure penalty.
        self.assertEqual(self.full_trials(), {4})
        self.assertEqual(storage.best_number, 4)

    def test_guard_failures_are_deleted_not_retained_as_milestones(self):
        storage = TrialStorage(self.runs, {'mode': 'milestones'})
        failed = self.finish(storage, 0, 1.e6,
            extra={'properties': [{'mean_value': .5, 'guard_failed': True}]})
        self.assertFalse(failed.exists())
        self.assertIsNone(storage.best_number)
        self.assertEqual(storage.milestones, set())
        self.finish(storage, 1, 10.)
        self.finish(storage, 2, 1.,  # Even an apparently better score cannot pass a failed guard.
            extra={'properties': [{'guard_failed': False}, {'guard_failed': True}]})
        self.assertEqual(self.full_trials(), {1})
        self.assertEqual(storage.best_number, 1)
        self.assertEqual(storage.milestones, {1})

    def test_keep_everything_includes_failed_and_guard_failed_runs(self):
        storage = TrialStorage(self.runs)
        self.finish(storage, 0, 1.e12, failed=True)
        self.finish(storage, 1, 1.e6, extra={'properties': [{'guard_failed': True}]})
        self.assertEqual(self.full_trials(), {0, 1})
        self.assertIsNone(storage.best_number)
        self.assertIsNone(storage.record_dir)

    def test_active_previous_run_and_source_directories_are_untouched(self):
        storage = TrialStorage(self.runs, {'mode': 'best'})
        untouched = []
        for name in ('trial_0042', 'trial_0100', 'source'):
            path = self.runs / name
            path.mkdir()
            file = path / 'cg.lammpstrj'
            file.write_text('untouched')
            untouched.append(file)
        self.finish(storage, 0, 10.)
        self.finish(storage, 1, 9.)
        self.assertTrue(all(p.read_text() == 'untouched' for p in untouched))
        with self.assertRaises(ValueError):
            storage._prune(42)

    def test_links_are_not_followed_and_protected_directory_is_rejected(self):
        storage = TrialStorage(self.runs, {'mode': 'best'})
        first = self.finish(storage, 0, 10.)
        # Model both a symlink and Windows junction via the common link check.
        from openmscg_gui import trial_storage as module
        original = module._linked
        protected_plot = first / 'table_plots'
        with patch.object(module, '_linked', side_effect=lambda p: p == protected_plot or original(p)):
            self.finish(storage, 1, 9.)
        self.assertTrue((protected_plot / 'pair.png').exists())
        self.assertTrue((first / 'cg.lammpstrj').exists())  # Whole-tree preflight precedes deletion.
        protected = TrialStorage(self.runs, {'mode': 'best'}, protected_paths=[first])
        with self.assertRaises(ValueError):
            protected._directory(0)

    def test_zero_threshold_zero_score_and_negative_scores(self):
        storage = TrialStorage(self.runs, {'mode': 'milestones', 'threshold_percent': 0})
        for number, score in enumerate([1., .5, .5, 0., -.1]):
            self.finish(storage, number, score)
        self.assertEqual(self.full_trials(), {0, 1, 3, 4})
        self.assertEqual(storage.milestones, {0, 1, 3, 4})

    def test_cleanup_failure_keeps_best_and_retries_later(self):
        storage = TrialStorage(self.runs, {'mode': 'best'})
        self.finish(storage, 0, 10.)
        with patch.object(storage, '_prune', side_effect=PermissionError('file in use')):
            self.finish(storage, 1, 9.)
        self.assertEqual(self.full_trials(), {0, 1})
        self.assertNotIn(0, storage.cleaned)
        self.finish(storage, 2, 11.)
        self.assertEqual(self.full_trials(), {1})

    def test_no_cleanup_if_audit_record_cannot_be_saved(self):
        storage = TrialStorage(self.runs, {'mode': 'best'})
        self.finish(storage, 0, 10.)
        write = Path.write_text

        def fail_record(path, *args, **kwargs):
            if path.name == 'trial_record.json':
                raise PermissionError('record not writable')
            return write(path, *args, **kwargs)

        with patch.object(Path, 'write_text', fail_record):
            with self.assertRaises(PermissionError):
                self.finish(storage, 1, 9.)
        self.assertEqual(self.full_trials(), {0, 1})
        self.assertEqual(storage.best_number, 0)

    def test_archive_failure_prevents_any_original_deletion(self):
        storage = TrialStorage(self.runs, {'mode': 'best'})
        first = self.finish(storage, 0, 10.)
        original = {str(p.relative_to(first)): p.read_bytes()
                    for p in first.rglob('*') if p.is_file()}
        with patch('openmscg_gui.trial_storage.shutil.copy2', side_effect=OSError('disk full')):
            self.finish(storage, 1, 9.)
        self.assertEqual({str(p.relative_to(first)): p.read_bytes()
                          for p in first.rglob('*') if p.is_file()}, original)
        self.assertNotIn(0, storage.cleaned)
        self.finish(storage, 2, 11.)  # Retry once archival storage is available again.
        self.assertFalse(first.exists())
        self.assertEqual(self.full_trials(), {1})

    def test_best_export_is_kept_once_in_cleanup_modes(self):
        for mode in ('best', 'milestones'):
            with self.subTest(mode=mode):
                runs = self.root / mode
                runs.mkdir()
                storage = TrialStorage(runs, {'mode': mode})
                original_runs = self.runs
                self.runs = runs
                self.finish(storage, 0, 10.)
                best = self.finish(storage, 1, 8.)
                destination = self.root / (mode + '_best')
                shutil.copytree(str(best), str(destination))
                storage.release_exported_best(destination)
                self.assertFalse(best.exists())
                self.assertEqual((destination / 'cg.lammpstrj').read_text(), 'test data')
                self.assertEqual(self.full_trials(), {0} if mode == 'milestones' else set())
                self.assertTrue((storage.archives[1] / 'trial_record.json').exists())
                self.assertTrue(json.loads((storage.archives[1] / 'storage.json').read_text())
                                ['exported_best'])
                self.runs = original_runs

    def test_incomplete_best_export_never_releases_original(self):
        storage = TrialStorage(self.runs, {'mode': 'best'})
        best = self.finish(storage, 0, 10.)
        destination = self.root / 'best'
        shutil.copytree(str(best), str(destination))
        (destination / 'cg.lammpstrj').write_text('incomplete')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            storage.release_exported_best(destination)
        self.assertTrue((best / 'cg.lammpstrj').exists())
        self.assertIsNone(storage.exported_best)

    def test_default_policy_keeps_original_after_best_export(self):
        storage = TrialStorage(self.runs)
        best = self.finish(storage, 0, 10.)
        destination = self.root / 'best'
        shutil.copytree(str(best), str(destination))
        storage.release_exported_best(destination)
        self.assertTrue(best.exists())

    def test_nested_logs_are_archived_before_removing_whole_run(self):
        storage = TrialStorage(self.runs, {'mode': 'best'})
        first = self.finish(storage, 0, 10.)
        nested = first / 'diagnostics' / 'engine'
        nested.mkdir(parents=True)
        (nested / 'log.lammps').write_text('nested diagnostic')
        self.finish(storage, 1, 9.)
        self.assertFalse(first.exists())
        self.assertEqual((storage.archives[0] / 'diagnostics' / 'engine' / 'log.lammps')
                         .read_text(), 'nested diagnostic')

    def test_archive_records_from_previous_studies_are_not_overwritten(self):
        first = TrialStorage(self.runs, {'mode': 'best'})
        self.finish(first, 0, 1.e12, failed=True)
        record = first.archives[0] / 'trial_record.json'
        content = record.read_bytes()
        second = TrialStorage(self.runs, {'mode': 'best'})
        self.finish(second, 0, 1.e12, failed=True)
        self.assertNotEqual(first.record_dir, second.record_dir)
        self.assertEqual(record.read_bytes(), content)

    def test_policy_validation_and_project_round_trip(self):
        self.assertEqual(normalize_trial_storage()['mode'], 'all')
        for values in ({'mode': 'unknown'}, {'threshold_percent': -1},
                       {'threshold_percent': 101}, {'threshold_percent': float('nan')}):
            with self.assertRaises(ValueError):
                normalize_trial_storage(values)
        self.assertEqual(SystemSpec.from_dict({}).trial_storage, {})
        system = SystemSpec(trial_storage={'mode': 'milestones', 'threshold_percent': 8.})
        self.assertEqual(SystemSpec.from_dict(asdict(system)).trial_storage, system.trial_storage)


if __name__ == '__main__':
    unittest.main()
