"""UI regression cases; run manually in a GUI dependency environment."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtWidgets
from openmscg_gui.workflow_views import DistributionPage


class DistributionEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        work = self.root / 'output' / 'work'
        work.mkdir(parents=True)
        histograms = self.root / 'output' / 'histograms'
        histograms.mkdir()
        self.entries = []
        for name in ('A-B', 'B-B'):
            histogram = histograms / ('histogram_xy_Bond-' + name + '.csv')
            histogram.write_text('x,y\n1,0.1\n2,1\n3,0.2\n', encoding='utf-8')
            self.entries.append({
                'name': 'Bond_' + name, 'kind': 'Bond', 'histogram_name': 'Bond-' + name,
                'histogram_csv': str(histogram),
                'fit_params': {'min': 1., 'max': 3., 'resolution': .1, 'order': 6, 'cutoff': 15.},
            })
        self.plan = work / 'fit_plan.json'
        self.original = json.dumps({'entries': self.entries})
        self.plan.write_text(self.original, encoding='utf-8')
        self.page = DistributionPage()
        self.addCleanup(self.page.deleteLater)
        self.changes = []
        self.page.overrideChanged.connect(lambda name, values: self.changes.append((name, values)))
        self.page.load_system_root(self.root, True)

    def test_manual_edit_and_reset_leave_generated_plan_untouched(self):
        self.assertEqual(self.changes, [])
        self.assertEqual(self.page.mode.currentText(), 'Automatic')
        self.page.mode.setCurrentIndex(1)
        self.page.spacing.setValue(.2)
        self.assertTrue(self.page._commit())
        self.assertEqual(self.changes[-1][1]['resolution'], .2)
        self.page._reset()
        self.assertEqual(self.changes[-1], ('Bond_A-B', None))
        self.assertEqual(self.page.spacing.value(), .1)
        self.assertEqual(self.page.mode.currentText(), 'Automatic')
        self.assertEqual(self.plan.read_text(encoding='utf-8'), self.original)

    def test_interaction_switch_restores_its_own_values(self):
        self.page.mode.setCurrentIndex(1)
        self.page.spacing.setValue(.2)
        self.page._commit()
        self.page.interactions.setCurrentRow(1)
        self.assertEqual(self.page.mode.currentText(), 'Automatic')
        self.assertEqual(self.page.spacing.value(), .1)
        self.page.interactions.setCurrentRow(0)
        self.assertEqual(self.page.mode.currentText(), 'Manual')
        self.assertEqual(self.page.spacing.value(), .2)

    def test_invalid_edit_does_not_emit_or_replace_saved_settings(self):
        self.page.mode.setCurrentIndex(1)
        before = list(self.changes)
        self.page.lower.setValue(4.)
        self.assertFalse(self.page._commit())
        self.assertEqual(self.changes, before)
        self.assertEqual(self.page.lower.value(), 1.)

    def test_drag_updates_preview_but_only_saves_on_release(self):
        self.page.plot.canvas.draw()
        axis = self.page._axis
        x, y = axis.transData.transform((1., .5))
        self.page._press(SimpleNamespace(button=1, inaxes=axis, x=x, y=y))
        x, y = axis.transData.transform((1.4, .5))
        self.page._motion(SimpleNamespace(x=x, y=y))
        self.assertEqual(self.changes, [])
        self.page._release(None)
        self.assertEqual(len(self.changes), 1)
        self.assertAlmostEqual(self.changes[0][1]['min'], 1.4, places=6)

    def test_automatic_tail_controls_reset_and_reference_lines(self):
        self.assertEqual(self.changes, [])
        self.assertEqual(self.page.spacing.value(), .1)  # No recalculation on loading legacy plans.
        self.page.tail_lower.setValue(25.)
        self.assertEqual(self.changes, [])  # Preview until editing finishes.
        self.assertTrue(self.page._commit())
        settings = self.changes[-1][1]
        self.assertEqual(settings['mode'], 'automatic')
        self.assertEqual(settings['lower_quantile'], .25)
        self.assertEqual(self.page.lower.value(), 2.)
        self.assertEqual(self.page.spacing.value(), .005)
        self.assertEqual([line.get_xdata()[0] for line in self.page._default_lines], [1., 3.])
        self.assertTrue(all(line.get_linestyle() == '--' for line in self.page._default_lines))
        self.page._reset()
        self.assertEqual(self.changes[-1], ('Bond_A-B', None))
        self.assertEqual(self.page.tail_lower.value(), .5)
        self.assertEqual(self.page.tail_upper.value(), 99.5)
        self.assertEqual(self.page.lower.value(), 1.)
        self.assertEqual(self.page.spacing.value(), .1)
        self.assertEqual(self.plan.read_text(encoding='utf-8'), self.original)

    def test_peak_mode_invalid_threshold_and_per_interaction_reload(self):
        self.page.auto_method.setCurrentIndex(1)
        self.page.peak_threshold.setValue(10.)
        self.assertTrue(self.page._commit())
        self.assertEqual(self.page.lower.value(), 2.)
        before = list(self.changes)
        self.page.peak_threshold.setValue(90.)  # Only one surviving bin: not a valid interval.
        self.assertFalse(self.page._commit())
        self.assertEqual(self.changes, before)
        self.assertEqual(self.page.peak_threshold.value(), 10.)
        saved = dict(self.page.overrides)
        self.page.interactions.setCurrentRow(1)
        self.assertEqual(self.page.auto_method.currentData(), 'quantile')
        self.assertEqual(self.page.spacing.value(), .1)
        self.page.load_system_root(self.root, True, saved)
        self.page.interactions.setCurrentRow(0)
        self.assertEqual(self.page.mode.currentText(), 'Automatic')
        self.assertEqual(self.page.auto_method.currentData(), 'threshold')
        self.assertEqual(self.page.peak_threshold.value(), 10.)
        self.assertEqual(self.page.lower.value(), 2.)
        self.assertEqual(self.changes, before)

    def test_order_only_keeps_legacy_automatic_bounds_and_spacing(self):
        self.page.order.setValue(4)
        self.assertTrue(self.page._commit())
        self.assertEqual(self.page.mode.currentText(), 'Automatic')
        self.assertEqual(self.page.spacing.value(), .1)
        self.assertEqual(self.changes[-1][1]['order'], 4)
        self.assertTrue(self.changes[-1][1]['original_bounds'])
        self.assertNotIn('resolution', self.changes[-1][1])
        self.page.interactions.setCurrentRow(1)
        self.assertEqual(self.page.order.value(), 6)
        self.page.interactions.setCurrentRow(0)
        self.assertEqual(self.page.order.value(), 4)
        self.assertEqual(self.page.spacing.value(), .1)
        self.page._reset()
        self.assertEqual(self.page.order.value(), 6)
        self.assertEqual(self.page.spacing.value(), .1)
        self.assertEqual(self.plan.read_text(encoding='utf-8'), self.original)

    def test_custom_spacing_survives_automatic_range_changes_and_reload(self):
        self.page.spacing_mode.setCurrentIndex(1)
        self.page.spacing.setValue(.2)
        self.page.order.setValue(4)
        self.assertTrue(self.page._commit())
        self.page.tail_lower.setValue(25)
        self.assertTrue(self.page._commit())
        self.assertEqual(self.page.lower.value(), 2)
        self.assertEqual(self.page.spacing.value(), .2)
        saved = dict(self.page.overrides)
        self.page.load_system_root(self.root, True, saved)
        self.assertEqual(self.page.mode.currentText(), 'Automatic')
        self.assertEqual(self.page.spacing_mode.currentIndex(), 1)
        self.assertEqual(self.page.spacing.value(), .2)
        self.assertEqual(self.page.order.value(), 4)
        self.page.spacing_mode.setCurrentIndex(0)
        self.assertEqual(self.page.spacing.value(), .005)
        self.assertEqual(self.page.order.value(), 4)
        self.page._reset()
        self.assertEqual(self.page.spacing.value(), .1)
        self.assertEqual(self.page.order.value(), 6)


if __name__ == '__main__':
    unittest.main()
