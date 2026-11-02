"""Orchestration regressions with a fake Optuna; no LAMMPS/app is launched."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openmscg_gui.optimization_history import active_study, best_directory, file_hash, read_json


class FakeTrial:
    def __init__(self, number):
        self.number, self.value = number, None
        self.state = SimpleNamespace(name='COMPLETE')
        self.params, self.distributions, self.user_attrs = {}, {}, {}
        self.datetime_start, self.datetime_complete = None, None


class FakeStudy:
    def __init__(self):
        self.trials = []

    def add_trial(self, trial):
        trial.number = len(self.trials)
        self.trials.append(trial)

    def optimize(self, objective, n_trials, callbacks):
        for _ in range(n_trials):
            trial = FakeTrial(len(self.trials))
            trial.value = objective(trial)
            self.trials.append(trial)
            for callback in callbacks:
                callback(self, trial)


class ExploreRefineContinuationTests(unittest.TestCase):
    def setUp(self):
        from openmscg_gui.backend.corrector import run, config
        self.run, self.config = run, config
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'input.data').write_text('fixed source input')
        (self.source / 'structure.data').write_text('fixed source structure')
        self.optimization = self.root / 'optimization'
        self.observed, self.studies = [], []
        self.reject_all = False
        def study(**kwargs):
            value = FakeStudy()
            self.studies.append(value)
            return value
        optuna = SimpleNamespace(create_study=study,
            samplers=SimpleNamespace(TPESampler=lambda **kwargs: None),
            distributions=SimpleNamespace(distribution_to_json=json.dumps, json_to_distribution=json.loads),
            trial=SimpleNamespace(TrialState=SimpleNamespace(COMPLETE=SimpleNamespace(name='COMPLETE'),
                FAIL=SimpleNamespace(name='FAIL')), create_trial=lambda **kwargs: SimpleNamespace(**kwargs)))
        patches = [patch.multiple(config, OPTIMIZATION_DIR=self.optimization,
            WORK_DIR=self.optimization / 'runs', BEST_DIR=self.optimization / 'best',
            SOURCE_LAMMPS_DIR=self.source, CONTINUE_RUN=False, ADD_EXPLORE=0, ADD_REFINE=0,
            SEARCH=dict(mode='explore_refine', explore_trials=3, refine_trials=2, baseline_first=True),
            TRIAL_STORAGE={'mode': 'all'}, INCLUDE_AA_CG_IN_LOSS=False),
            patch.multiple(run, HISTORY=None, OPTUNA=None, ACTIVE_PHASE=None, SEARCH_CONTEXT=None),
            patch.object(run, '_import_optuna', return_value=optuna),
            patch.object(run, '_require_source_lammps'),
            patch.object(run, '_scientific_settings', return_value={'protocol': 'fixed'}),
            patch.object(run, '_initial_gaussians', return_value={'Pair_A-A.table': dict(amplitude=0., center=4., sigma=1.)}),
            patch.object(run, 'objective', side_effect=self.objective),
            patch.object(run.subprocess, 'run', side_effect=AssertionError('No simulations in this test'))]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def objective(self, trial):
        run = self.run
        self.observed.append((trial.number, run.ACTIVE_PHASE, run.SEARCH_CONTEXT.best_number))
        run.HISTORY.reserve(trial.number, run.ACTIVE_PHASE)
        folder = run._trial_dir(trial.number)
        folder.mkdir()
        score = [10., 7., 8., 6., 9., 5., 4., 8.][trial.number]
        trial.user_attrs.update(phase=run.ACTIVE_PHASE, baseline=trial.number == 0)
        metadata = dict(pair_corrections=[dict(table='Pair_A-A.table', amplitude=trial.number * .1,
                                              center=4., sigma=1.)])
        (folder / 'correction.json').write_text(json.dumps(metadata))
        (folder / 'score.json').write_text(json.dumps(dict(score=score, properties=[{'guard_failed': self.reject_all}])))
        (folder / 'input.data').write_text('trial input')
        (folder / 'cg.lammpstrj').write_text('trial {}'.format(trial.number))
        return score

    def test_explore_refine_then_continue_appends_and_preserves_old_best(self):
        self.run.main()
        self.assertEqual([phase for _, phase, _ in self.observed], ['explore'] * 3 + ['refine'] * 2)
        self.assertEqual(self.observed[3][2], 1)  # Refine starts from Explore's best, not zero.
        session = active_study(self.optimization)
        first_best = best_directory(self.optimization)
        self.assertEqual(first_best.name, 'trial_0003')
        before = {str(p.relative_to(session)): file_hash(p)
                  for folder in ('history', 'runs', 'best', 'source')
                  for p in (session / folder).rglob('*') if p.is_file()}
        self.config.SOURCE_LAMMPS_DIR = self.source
        self.config.CONTINUE_RUN = True
        self.config.ADD_EXPLORE, self.config.ADD_REFINE = 1, 2
        self.run.main()
        self.assertEqual(active_study(self.optimization), session)
        self.assertEqual([number for number, _, _ in self.observed], list(range(8)))
        self.assertEqual([phase for _, phase, _ in self.observed[-3:]], ['explore', 'refine', 'refine'])
        self.assertEqual(self.observed[6][2], 5)
        self.assertEqual(best_directory(self.optimization).name, 'trial_0006')
        self.assertEqual((first_best / 'cg.lammpstrj').read_text(), 'trial 3')
        for relative, digest in before.items():
            self.assertEqual(file_hash(session / relative), digest, relative)
        self.assertEqual(len(self.studies[-1].trials), 8)

    def test_new_run_uses_a_separate_study_instead_of_reusing_folders(self):
        self.run.main()
        first = active_study(self.optimization)
        before = {str(p.relative_to(first)): file_hash(p) for p in first.rglob('*') if p.is_file()}
        self.config.SOURCE_LAMMPS_DIR = self.source
        self.run.main()
        self.assertNotEqual(first, active_study(self.optimization))
        for relative, digest in before.items():
            self.assertEqual(file_hash(first / relative), digest)

    def test_no_valid_explore_candidate_skips_refine_but_history_is_resumable(self):
        self.reject_all = True
        with self.assertRaisesRegex(RuntimeError, 'No trial completed'):
            self.run.main()
        self.assertEqual(len(self.observed), 3)
        session = active_study(self.optimization)
        self.assertEqual(len(list((session / 'history').glob('trial_*.json'))), 3)
        self.assertIsNone(read_json(session / 'state.json')['best_number'])


if __name__ == '__main__':
    unittest.main()
