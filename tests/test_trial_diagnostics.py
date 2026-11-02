"""Saved-result diagnostics; no process is started."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openmscg_gui.trial_diagnostics import lammps_failure, log_tail, load_trial_results, outcome_reason


class TrialDiagnosticsTests(unittest.TestCase):
    def test_cutoff_error_reports_pair_not_bond_and_preserves_original_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            message = 'ERROR on proc 3: Pair distance < table inner cutoff: ijtype 1 2 dist 0.49'
            (root / 'log').write_text(message + '\nMPI_ABORT invoked\n')
            (root / 'structure.data').write_text('Example\n\nMasses\n\n1 15 # A\n2 14 # B\n\nAtoms\n')
            data = lammps_failure(root / 'log', root / 'structure.data', 1)
            self.assertEqual(data['interaction'], 'A–B')
            self.assertEqual(data['lammps_message'], message)
            self.assertEqual(data['failure_category'], 'pair_inner_cutoff')
            self.assertNotIn('repulsion too weak', data['failure_reason'])

    def test_missing_label_and_non_lammps_exit_have_safe_fallbacks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = lammps_failure(root / 'missing.log', root / 'missing.data', 127)
            self.assertIn('127', data['failure_reason'])
            self.assertEqual(data['score'], 1.e12)

    def test_log_read_is_bounded(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'log'
            path.write_text('x' * 1000 + 'ending')
            text = log_tail(path, limit=20)
            self.assertTrue(text.endswith('ending'))
            self.assertLess(len(text), 60)

    def test_archived_trials_are_listed_and_duplicate_best_record_is_not_counted_twice(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            record = dict(trial=0, started='2026-10-28 10:00:00', user_attributes={'baseline': True})
            for folder in ('best', 'trial_records/run_example/trial_0000'):
                path = root / folder
                path.mkdir(parents=True)
                (path / 'trial_record.json').write_text(json.dumps(record))
                (path / 'score.json').write_text(json.dumps({'score': 1.}))
            failed = root / 'trial_records/run_example/trial_0001'
            failed.mkdir()
            (failed / 'trial_record.json').write_text(json.dumps(dict(trial=1, started='2026-10-28 11:00:00')))
            (failed / 'score.json').write_text(json.dumps(dict(score=1.e12, failed=True,
                                                             failure_reason='Example failure')))
            rows = load_trial_results(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['kind'], 'Baseline')
            self.assertEqual(rows[0]['directory'], root / 'best')
            self.assertEqual(rows[1]['reason'], 'Example failure')
            self.assertEqual(rows[1]['outcome'], 'Rejected')

    def test_property_guard_is_described_without_calling_it_a_crash(self):
        reason = outcome_reason({'score': 1.e6, 'properties': [{'column': 'density', 'guard_failed': True}]})
        self.assertIn('density', reason)
        self.assertIn('limits', reason)


if __name__ == '__main__':
    unittest.main()
