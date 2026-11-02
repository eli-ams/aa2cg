"""Search regressions; no simulations or Optuna execution required."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openmscg_gui.optimization_search import IncrementalSearch, normalize_search, valid_score
from openmscg_gui.project import SystemSpec


class FakeTrial:
    def __init__(self, number, step=1.):
        self.number, self.step = number, step
        self.params, self.user_attrs = {}, {}

    def suggest_float(self, name, low, high):
        self.params[name] = self.step
        return self.step

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class IncrementalSearchTests(unittest.TestCase):
    def setUp(self):
        self.initial = {'Pair_A-A.table': dict(amplitude=0., center=4., sigma=1.),
                        'Pair_A-B.table': dict(amplitude=0., center=5., sigma=1.)}
        self.bounds = dict(amplitude=(-2., 2.), center=(2.5, 8.), sigma=(.2, 2.))
        self.search = IncrementalSearch(self.initial, self.bounds)

    def test_proposal_changes_one_coordinate_with_fixed_relative_optuna_bounds(self):
        trial = FakeTrial(1)
        values, details = self.search.propose(trial)
        self.assertAlmostEqual(values['Pair_A-A.table']['amplitude'], .4)
        self.assertEqual(values['Pair_A-B.table'], self.initial['Pair_A-B.table'])
        self.assertEqual(details['changed_parameter'], 'amplitude')
        self.assertIsNone(details['reference_trial'])
        self.assertEqual(self.search.reference, self.initial)

    def test_only_better_valid_trials_move_reference_and_input_is_copied(self):
        values, _ = self.search.propose(FakeTrial(1))
        self.assertTrue(self.search.accept(1, values, 10.))
        original = deepcopy(self.search.reference)
        values['Pair_A-A.table']['amplitude'] = -99.
        self.assertEqual(self.search.reference, original)
        for score in (None, float('nan'), 11., 10.):
            self.assertFalse(self.search.accept(2, self.initial, score))
        self.assertEqual(self.search.reference, original)
        next_values, details = self.search.propose(FakeTrial(2))
        self.assertEqual(details['reference_trial'], 1)
        self.assertEqual(next_values['Pair_A-A.table'], original['Pair_A-A.table'])
        self.assertTrue(self.search.accept(2, next_values, 9.))

    def test_center_and_width_are_explored_after_amplitudes_and_stay_bounded(self):
        values = deepcopy(self.initial)
        values['Pair_A-A.table']['amplitude'] = 2.
        self.search.accept(0, values, 0.)
        self.search.proposals = 2
        proposal, details = self.search.propose(FakeTrial(3))
        self.assertEqual(details['changed_parameter'], 'center')
        self.assertAlmostEqual(proposal['Pair_A-A.table']['center'], 4.55)
        self.search.proposals = 4
        proposal, details = self.search.propose(FakeTrial(5))
        self.assertEqual(details['changed_parameter'], 'sigma')
        self.assertAlmostEqual(proposal['Pair_A-A.table']['sigma'], 1.18)
        self.search.proposals = 0
        proposal, _ = self.search.propose(FakeTrial(1))
        self.assertEqual(proposal['Pair_A-A.table']['amplitude'], 2.)

    def test_inactive_gaussian_uses_amplitude_instead_of_wasting_a_width_trial(self):
        self.search.proposals = 4
        proposal, details = self.search.propose(FakeTrial(5))
        self.assertEqual(details['changed_parameter'], 'amplitude')
        self.assertNotEqual(proposal['Pair_A-A.table']['amplitude'], 0.)

    def test_guard_failures_never_become_reference_candidates(self):
        for data in ({'score': 1., 'failed': True}, {'score': 1., 'properties': [{'guard_failed': True}]},
                     {'score': 1., 'properties': [{'mean_value': float('nan')}]},
                     {'score': float('inf')}, {}):
            self.assertIsNone(valid_score(data))
        self.assertEqual(valid_score({'score': 0., 'properties': [{'guard_failed': False}]}), 0.)

    def test_defaults_validation_and_per_system_roundtrip(self):
        self.assertEqual(normalize_search()['mode'], 'explore_refine')
        self.assertEqual(normalize_search()['explore_trials'], 30)
        self.assertEqual(normalize_search()['refine_trials'], 20)
        self.assertTrue(normalize_search()['baseline_first'])
        for values in ({'mode': 'unknown'}, {'step_percent': 0}, {'step_percent': float('nan')},
                       {'step_percent': 101}, {'baseline_first': 'yes'}):
            with self.assertRaises(ValueError):
                normalize_search(values)
        system = SystemSpec(optimization_search=dict(mode='incremental', baseline_first=True, step_percent=5.))
        self.assertEqual(SystemSpec.from_dict(asdict(system)).optimization_search, system.optimization_search)
        self.assertEqual(SystemSpec.from_dict({}).optimization_search, {})


class OptimizerIntegrationTests(unittest.TestCase):
    def test_reference_updates_before_cleanup_and_never_follows_bad_trials(self):
        from openmscg_gui.backend.corrector import run, config
        initial = {'Pair_A-A.table': dict(amplitude=0., center=4., sigma=1.)}
        context = IncrementalSearch(initial, dict(amplitude=(-2., 2.), center=(2.5, 8.), sigma=(.2, 2.)))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            references = []
            def storage(study, trial):
                references.append(context.best_number)
            with patch.object(config, 'WORK_DIR', root), patch.object(run, 'SEARCH_CONTEXT', context):
                for number, (score, guards) in enumerate(((10., False), (9., False), (1., True), (11., False))):
                    trial_dir = root / 'trial_{:04d}'.format(number)
                    trial_dir.mkdir()
                    (trial_dir / 'score.json').write_text(json.dumps(dict(score=score,
                        properties=[{'guard_failed': guards}])))
                    (trial_dir / 'correction.json').write_text(json.dumps(dict(pair_corrections=[
                        dict(table='Pair_A-A.table', amplitude=number * .1, center=4., sigma=1.)])))
                    trial = SimpleNamespace(number=number, value=score, state=SimpleNamespace(name='COMPLETE'))
                    run._completed_trial(None, trial, storage)
            self.assertEqual(references, [0, 1, 1, 1])
            self.assertEqual(context.reference['Pair_A-A.table']['amplitude'], .1)

    def test_baseline_preserves_both_energy_and_force_columns_byte_for_byte(self):
        from openmscg_gui.backend.corrector import run, config
        from openmscg_gui.simulation_protocol import SMOKE_MARKER, DEFAULT_PROTOCOL, run_lines
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'input.data').write_text('units real\n' + SMOKE_MARKER + '\n' +
                '\n'.join(run_lines(DEFAULT_PROTOCOL, 'id type x y z')))
            raw = '# original forces intentionally differ from numerical energy gradient\n\nA-A\nN 3\n\n1 3 2 90\n2 4 -1 80\n3 5 0 70\n'
            (source / 'Pair_A-A.table').write_text(raw)
            trial = FakeTrial(0)
            with patch.multiple(config, SOURCE_LAMMPS_DIR=source, WORK_DIR=root / 'runs',
                    SIMULATION_PROTOCOL={}, SEARCH={'baseline_first': True}), \
                 patch.object(run, '_run_and_score', return_value=1.), \
                 patch.object(run, '_suggest_and_apply_pair_corrections', side_effect=AssertionError('Baseline modified')), \
                 patch.object(run.subprocess, 'run', side_effect=AssertionError('No simulation')):
                run.objective(trial)
            target = root / 'runs' / 'trial_0000'
            self.assertEqual((source / 'Pair_A-A.table').read_bytes(), (target / 'Pair_A-A.table').read_bytes())
            self.assertTrue(trial.user_attrs['baseline'])
            metadata = json.loads((target / 'correction.json').read_text())
            self.assertEqual(metadata['pair_corrections'][0]['amplitude'], 0.)

    def test_lammps_failure_reason_is_written_to_score(self):
        from openmscg_gui.backend.corrector import run
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'structure.data').write_text('Example\n\nMasses\n\n1 15 # A\n\nAtoms\n')
            def engine(*args, **kwargs):
                kwargs['stdout'].write('ERROR: Pair distance < table inner cutoff: ijtype 1 1 dist 0.49\n')
                return SimpleNamespace(returncode=1)
            with patch.object(run.subprocess, 'run', side_effect=engine):
                self.assertEqual(run._run_and_score(root), 1.e12)
            data = json.loads((root / 'score.json').read_text())
            self.assertEqual(data['interaction'], 'A–A')
            self.assertEqual(data['failure_category'], 'pair_inner_cutoff')

    def test_missing_property_output_is_a_recorded_failure_not_a_broken_study(self):
        from openmscg_gui.backend.corrector import run
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(run.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                 patch.object(run, '_score_trial', side_effect=ValueError('No production records')):
                self.assertEqual(run._run_and_score(root), 1.e12)
            data = json.loads((root / 'score.json').read_text())
            self.assertEqual(data['failure_category'], 'scoring_error')
            self.assertIn('No production records', data['failure_reason'])


if __name__ == '__main__':
    unittest.main()
