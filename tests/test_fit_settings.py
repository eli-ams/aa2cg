"""Regression cases for fitting overrides. No OpenMSCG execution required."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openmscg_gui.fit_settings import (effective_fit_plan, validate_override,
                                       automatic_defaults, normalize_override, resolve_override)
from openmscg_gui.project import Project, SystemSpec


class FitSettingsTests(unittest.TestCase):
    def setUp(self):
        self.entry = {
            'name': 'Pair_A-B', 'kind': 'Pair', 'interaction': ['A', 'B'],
            'fit_params': {'min': 2.4, 'max': 10., 'resolution': .05,
                           'order': 6, 'model': 'BSpline', 'cutoff': 15.},
        }
        self.override = {'min': 3., 'max': 9., 'resolution': .1}

    def test_no_override_preserves_automatic_plan_exactly(self):
        plan = {'entries': [deepcopy(self.entry)], 'metadata': {'example': 1}}
        result = effective_fit_plan(plan, {})
        self.assertEqual(plan, result)
        self.assertIsNot(plan['entries'][0], result['entries'][0])

    def test_override_changes_only_selected_parameters_and_interaction(self):
        other = deepcopy(self.entry)
        other['name'] = 'Pair_B-B'
        plan = {'entries': [deepcopy(self.entry), other]}
        before = deepcopy(plan)
        result = effective_fit_plan(plan, {'Pair_A-B': self.override})
        self.assertEqual(plan, before)
        expected = dict(self.entry['fit_params'], **self.override)
        self.assertEqual(result['entries'][0]['fit_params'], expected)
        self.assertEqual(result['entries'][1], other)
        self.assertEqual(effective_fit_plan(plan, {}), before)  # Reset uses the original.

    def test_unmatched_saved_overrides_do_not_affect_new_interactions(self):
        plan = {'entries': [self.entry]}
        self.assertEqual(effective_fit_plan(plan, {'Bond_C-D': self.override}), plan)

    def test_invalid_bounds_spacing_and_cutoffs_are_rejected(self):
        for changes in ({'min': -1}, {'min': 9}, {'max': 16}, {'max': float('inf')},
                        {'resolution': 0}, {'resolution': float('nan')},
                        {'resolution': 7}, {'resolution': 1e-8}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    validate_override(self.entry, dict(self.override, **changes))

    def test_angles_are_degrees_and_not_limited_by_pair_neighbor_cutoff(self):
        angle = dict(self.entry, kind='Angle', name='Angle_A-B-A')
        values = {'min': 20, 'max': 170, 'resolution': 1}
        self.assertEqual(validate_override(angle, values), values)
        with self.assertRaises(ValueError):
            validate_override(angle, dict(values, max=181))

    def test_dihedral_defaults_are_untouched(self):
        entry = dict(self.entry, kind='Dihedral', name='Dihedral_A-B-B-A')
        plan = {'entries': [entry]}
        self.assertEqual(effective_fit_plan(plan, {}), plan)
        with self.assertRaises(ValueError):
            effective_fit_plan(plan, {entry['name']: self.override})

    def test_automatic_modes_use_histogram_weight_and_relative_peak(self):
        samples = [(1., .1), (2., 1.), (3., .2)]
        settings = dict(automatic_defaults(self.entry), lower_quantile=.25)
        expected = {'min': 2., 'max': 3., 'resolution': .005}
        self.assertEqual(resolve_override(self.entry, settings, samples=samples), expected)
        settings.update(method='threshold', histogram_threshold_percent=10.)
        self.assertEqual(resolve_override(self.entry, settings, samples=samples), expected)
        # The first bin equals the threshold: backend semantics use strictly above.
        self.assertEqual(normalize_override(self.entry, settings), settings)

    def test_automatic_uses_original_interval_count_and_rounding(self):
        entry = deepcopy(self.entry)
        entry['auto_fit_settings'] = {'mode': 'threshold', 'bins': 100,
                                      'histogram_threshold_percent': 2.}
        settings = automatic_defaults(entry)
        self.assertEqual(settings['method'], 'threshold')
        self.assertEqual(settings['histogram_threshold_percent'], 2.)
        result = resolve_override(entry, settings, samples=[(1.23456, 1), (3.45678, 1)])
        self.assertEqual(result, {'min': 1.235, 'max': 3.457, 'resolution': .02222})

    def test_invalid_automatic_ranges_do_not_fall_back_silently(self):
        settings = automatic_defaults(self.entry)
        for change in ({'lower_quantile': .999}, {'histogram_threshold_percent': 100},
                       {'upper_quantile': float('nan')}, {'method': 'unknown'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_override(self.entry, dict(settings, **change))
        for samples in ([], [(1, 0), (2, 0)], [(1, 1)]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                resolve_override(self.entry, settings, samples=samples)
        with self.assertRaises(ValueError):
            resolve_override(self.entry, dict(settings, method='threshold',
                             histogram_threshold_percent=90), samples=[(1, .1), (2, 1), (3, .2)])

    def test_worker_resolution_matches_preview_without_changing_baseline(self):
        with TemporaryDirectory() as directory:
            histogram_dir = Path(directory) / 'output' / 'histograms'
            histogram_dir.mkdir(parents=True)
            (histogram_dir / 'histogram_xy_Pair-A-B.csv').write_text(
                'x,y\n1,0.1\n2,1\n3,0.2\n', encoding='utf-8')
            entry = dict(self.entry, histogram_name='Pair-A-B', histogram_csv='old/location.csv')
            plan = {'entries': [entry]}
            before = deepcopy(plan)
            settings = dict(automatic_defaults(entry), lower_quantile=.25)
            result = effective_fit_plan(plan, {entry['name']: settings}, root=directory)
            preview = resolve_override(entry, settings, samples=[(1, .1), (2, 1), (3, .2)])
            self.assertEqual(result['entries'][0]['fit_params'], dict(entry['fit_params'], **preview))
            self.assertEqual(plan, before)
            self.assertEqual(effective_fit_plan(plan, {}), before)

    def test_old_projects_and_saved_settings_round_trip(self):
        legacy = SystemSpec.from_dict({'name': 'Legacy'})
        self.assertEqual(legacy.fit_overrides, {})
        self.assertEqual(legacy.invalidated_stages, [])
        with TemporaryDirectory() as directory:
            project = Project.create(directory)
            system = project.add_system('Example')
            system.fit_overrides[self.entry['name']] = self.override
            system.fit_overrides['Bond_A-A'] = dict(automatic_defaults(self.entry), lower_quantile=.1,
                                                   resolution=.03, order=4)
            system.invalidated_stages = ['force', 'postprocess']
            project.merge_invalidated = True
            project.save()
            loaded = Project.load(Path(directory))
            self.assertEqual(loaded.system(system.id).fit_overrides, system.fit_overrides)
            self.assertEqual(loaded.system(system.id).invalidated_stages, system.invalidated_stages)
            self.assertTrue(loaded.merge_invalidated)
            system.fit_overrides.clear()
            project.save()
            self.assertEqual(Project.load(directory).system(system.id).fit_overrides, {})

    def test_custom_spline_settings_are_independent_of_automatic_bounds(self):
        settings = dict(automatic_defaults(self.entry), lower_quantile=.25, resolution=.1, order=4)
        result = resolve_override(self.entry, settings, samples=[(1, .1), (2, 1), (3, .2)])
        self.assertEqual(result, {'min': 2., 'max': 3., 'resolution': .1, 'order': 4})
        settings['lower_quantile'] = .005
        self.assertEqual(resolve_override(self.entry, settings, samples=[(1, .1), (2, 1), (3, .2)]),
                         {'min': 1., 'max': 3., 'resolution': .1, 'order': 4})

    def test_order_only_edit_preserves_exact_original_bounds_and_spacing(self):
        settings = dict(automatic_defaults(self.entry), original_bounds=True, order=4)
        # No histogram is needed: no bounds or spacing were edited.
        result = resolve_override(self.entry, settings)
        self.assertEqual(result, {'min': 2.4, 'max': 10., 'resolution': .05, 'order': 4})
        plan = {'entries': [self.entry]}
        fitted = effective_fit_plan(plan, {self.entry['name']: settings})
        self.assertEqual(fitted['entries'][0]['fit_params']['order'], 4)
        self.assertEqual(plan['entries'][0]['fit_params']['order'], 6)
        from openmscg_gui.backend.openmscg_wrap.fitting import cgfm_spec
        self.assertIn('order=4', cgfm_spec(fitted['entries'][0]['fit_params'], ('A', 'B')))

    def test_invalid_orders_and_automatic_spacing_are_rejected(self):
        for order in (0, 1, 7, 3.5, float('nan'), 'bad'):
            with self.subTest(order=order), self.assertRaises(ValueError):
                normalize_override(self.entry, dict(self.override, order=order))
        with self.assertRaises(ValueError):
            resolve_override(self.entry, dict(automatic_defaults(self.entry), resolution=2),
                             samples=[(1, .1), (2, 1)])

    def test_manual_spacing_ui_metadata_does_not_enter_backend_plan(self):
        override = dict(self.override, order=4, spacing_mode='custom')
        self.assertEqual(normalize_override(self.entry, override), override)
        resolved = resolve_override(self.entry, override)
        self.assertNotIn('spacing_mode', resolved)
        self.assertEqual(resolved['order'], 4)

    def test_dihedral_spline_changes_keep_fixed_periodic_domain(self):
        entry = deepcopy(self.entry)
        entry.update(kind='Dihedral', name='Dihedral_A-B-B-A')
        entry['fit_params'].update(min=-180., max=180., resolution=5.)
        settings = dict(automatic_defaults(entry), original_bounds=True, resolution=3., order=4)
        self.assertEqual(resolve_override(entry, settings),
                         {'min': -180., 'max': 180., 'resolution': 3., 'order': 4})
        with self.assertRaises(ValueError):
            normalize_override(entry, dict(settings, original_bounds=False))


if __name__ == '__main__':
    unittest.main()
