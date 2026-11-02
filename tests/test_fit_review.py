"""Fit review regression cases; run manually with GUI dependencies."""
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets
from openmscg_gui.fit_review import FitReview
from openmscg_gui.fit_settings import automatic_defaults


class FitReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / 'output' / 'work'
        self.work.mkdir(parents=True)
        tables = self.root / 'output' / 'tables'
        tables.mkdir()
        self.raw = tables / 'Bond_A-B.table'
        self.raw.write_text('# raw\n\nBond_A-B\nN 3\n\n1 1 2 -3\n2 2 0 0\n3 3 2 3\n', encoding='utf-8')
        self.raw_original = self.raw.read_bytes()
        self.entry = {'name': 'Bond_A-B', 'kind': 'Bond',
                      'fit_params': {'min': 1., 'max': 3., 'resolution': .1,
                                     'order': 6, 'cutoff': 15., 'model': 'BSpline'}}
        self.plan = {'entries': [self.entry]}
        (self.work / 'fit_plan.json').write_text(json.dumps(self.plan), encoding='utf-8')
        self.record = {'method': 'independent', 'fit_overrides': {}, 'fit_plan': self.plan}
        (self.work / 'force_match.json').write_text(json.dumps(self.record), encoding='utf-8')
        self.review = FitReview()
        self.addCleanup(self.review.deleteLater)

    def test_current_raw_results_and_requested_settings(self):
        self.review.load(self.root, {}, True, 'complete', 'independent')
        item = self.review.settings.topLevelItem(0)
        self.assertEqual([item.text(i) for i in (0, 2, 3, 4)], ['Bond_A-B', '0.1', '6', 'Original'])
        self.assertTrue(self.review._is_current())
        self.assertEqual(len(self.review.plot.figure.axes), 2)
        self.assertEqual(list(self.review.plot.figure.axes[0].lines[0].get_ydata()), [2., 0., 2.])
        self.assertEqual(list(self.review.plot.figure.axes[1].lines[0].get_ydata()), [-3., 0., 3.])
        self.assertEqual(self.raw.read_bytes(), self.raw_original)

    def test_custom_order_and_method_mark_old_curves_stale_without_replacing_them(self):
        override = dict(automatic_defaults(self.entry), original_bounds=True, order=4)
        self.review.load(self.root, {'Bond_A-B': override}, True, 'pending', 'independent')
        self.assertFalse(self.review._is_current())
        self.assertEqual(self.review.settings.topLevelItem(0).text(3), '4')
        self.assertIn('Previous', self.review.plot.figure.axes[0].get_title())
        self.assertEqual(list(self.review.plot.figure.axes[0].lines[0].get_ydata()), [2., 0., 2.])
        self.review.load(self.root, {}, True, 'complete', 'independent')
        self.review.set_method('coupled')
        self.assertFalse(self.review._is_current())
        self.review.set_method('independent')
        self.assertTrue(self.review._is_current())
        self.assertEqual(self.raw.read_bytes(), self.raw_original)

    def test_failed_fit_and_changed_plan_are_not_current(self):
        self.review.load(self.root, {}, True, 'failed', 'independent')
        self.assertFalse(self.review._is_current())
        changed = deepcopy(self.plan)
        changed['entries'][0]['fit_params']['order'] = 4
        (self.work / 'fit_plan.json').write_text(json.dumps(changed), encoding='utf-8')
        self.review.load(self.root, {}, True, 'complete', 'independent')
        self.assertFalse(self.review._is_current())

    def test_missing_result_and_empty_system_are_safe(self):
        self.raw.unlink()
        self.review.load(self.root, {}, True, 'pending', 'independent')
        self.assertIn('No raw table', self.review.detail.text())
        self.review.load(None)
        self.assertEqual(self.review.settings.topLevelItemCount(), 0)
        self.assertFalse(self.review._is_current())


if __name__ == '__main__':
    unittest.main()
