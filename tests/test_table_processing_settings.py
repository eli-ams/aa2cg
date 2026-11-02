"""Processing regression cases; intentionally not run by the implementation task."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from openmscg_gui.backend.table_processing import (
    TableProcessor, TRIM_TRIALS, SMOOTH_WINDOWS, FIT_POINTS,
    default_processing_settings, normalize_processing_settings, processing_attempts, preview_table,
)
from openmscg_gui.project import SystemSpec, Project
from openmscg_gui.backend import table_processing


class TableProcessingSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'Bond_A-B.table'
        self.x = np.linspace(1., 3., 161)
        self.path.write_text('# synthetic\n\nBond_A-B\nN 161\n\n' + ''.join(
            '{} {:.12g} {:.12g} {:.12g}\n'.format(i + 1, x, (x - 2) ** 2, -2 * (x - 2))
            for i, x in enumerate(self.x)), encoding='utf-8')
        self.original = self.path.read_bytes()

    def test_default_search_matches_original_order_and_first_accepted_result(self):
        reference, chosen = None, None
        for trim in TRIM_TRIALS:
            for smooth in SMOOTH_WINDOWS:
                for fit in FIT_POINTS:
                    reference = TableProcessor(self.path.name, str(self.root), '')
                    reference.run(trim, fit, smooth)
                    chosen = (trim, smooth, fit)
                    if reference.overall_accept:
                        break
                if reference.overall_accept:
                    break
            if reference.overall_accept:
                break
        preview = preview_table(self.path)
        self.assertEqual(tuple(preview['recipe'][k] for k in ('trim', 'smooth', 'fit_points')), chosen)
        self.assertEqual(preview['accepted'], bool(reference.overall_accept))
        for actual, expected in zip(preview['curves'], (reference.distances, reference.energies, reference.forces)):
            np.testing.assert_allclose(actual, expected)

    def test_custom_preview_and_export_calculation_match_without_writing(self):
        settings = dict(trim=.05, smooth=7, fit_points=5, bounds=[.4, 4.5])
        with patch.object(TableProcessor, 'export_files', side_effect=AssertionError('Preview exported')):
            preview = preview_table(self.path, settings)
        attempts = list(processing_attempts(self.path.name, str(self.root), str(self.root / 'unused'), settings))
        self.assertEqual(len(attempts), 1)
        processor, recipe = attempts[0]
        self.assertEqual(recipe, preview['recipe'])
        for actual, expected in zip(preview['curves'], (processor.distances, processor.energies, processor.forces)):
            np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(preview['curves'][0][0], .4)
        self.assertAlmostEqual(preview['curves'][0][-1], 4.5)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_invalid_recipes_are_rejected_and_dihedral_domain_is_fixed(self):
        defaults = default_processing_settings('Bond')
        for change in ({'smooth': 6}, {'trim': .5}, {'bounds': [4, 1]},
                       {'smooth': 1001}, {'fit_points': 1000}, {'trim': float('nan')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_processing_settings('Bond', dict(defaults, **change), self.x)
        periodic = normalize_processing_settings('Dihedral', dict(defaults, bounds=[.5, 4]), self.x)
        self.assertEqual(periodic['trim'], 0)
        self.assertEqual(periodic['bounds'], [-180., 180.])

    def test_cancelled_preview_does_not_write(self):
        with self.assertRaises(InterruptedError):
            preview_table(self.path, cancelled=lambda: True)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_invalid_custom_recipe_is_checked_before_export_cleanup(self):
        with patch.object(table_processing, 'RAW_TABLE_DIR', self.root), \
                patch.object(table_processing, 'PROCESSED_TABLE_DIR', self.root / 'processed'), \
                patch.object(table_processing, 'delete_all_files_in_directory') as cleanup:
            with self.assertRaises(ValueError):
                table_processing.main({self.path.name: {'smooth': 6}})
            cleanup.assert_not_called()
        self.assertFalse((self.root / 'processed').exists())
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_project_round_trip_and_legacy_default(self):
        self.assertEqual(SystemSpec.from_dict({'name': 'Legacy'}).table_overrides, {})
        project = Project.create(self.root / 'project')
        system = project.add_system('Example')
        system.table_overrides['Bond_A-B.table'] = normalize_processing_settings(
            'Bond', default_processing_settings('Bond'), self.x)
        project.save()
        self.assertEqual(Project.load(project.root).system(system.id).table_overrides, system.table_overrides)


if __name__ == '__main__':
    unittest.main()
