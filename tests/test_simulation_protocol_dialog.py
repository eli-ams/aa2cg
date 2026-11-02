"""Dialog regression cases; run manually with GUI dependencies."""
import os
from copy import deepcopy
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets
from openmscg_gui.optimization import property_settings
from openmscg_gui.simulation_protocol import DEFAULT_PROTOCOL, SMOKE_MARKER, run_lines
from openmscg_gui.simulation_protocol_dialog import SimulationProtocolDialog


class SimulationProtocolDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_dialog(self):
        self.properties = [property_settings('v_sysdensity')]
        text = 'units real\n' + SMOKE_MARKER + '\n' + '\n'.join(run_lines(DEFAULT_PROTOCOL, 'id type x y z'))
        dialog = SimulationProtocolDialog({}, self.properties, text)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_inherited_default_and_duration_preview(self):
        dialog = self.make_dialog()
        self.assertFalse(dialog.custom.isChecked())
        self.assertFalse(dialog.fields_box.isEnabled())
        self.assertIn('sample 500 ps', dialog.summary.text())
        dialog.custom.setChecked(True)
        dialog.fields['production_steps'].setValue(2_000_000)
        self.assertIn('sample 2 ns', dialog.summary.text())
        dialog._accept()
        self.assertEqual(dialog.overrides['production_steps'], 2_000_000)

    def test_compact_layout_keeps_editors_at_their_natural_height(self):
        dialog = self.make_dialog()
        columns = dialog.layout().itemAt(1).layout()
        self.assertIsInstance(columns, QtWidgets.QHBoxLayout)
        self.assertIs(columns.itemAt(0).widget(), dialog.ramp_box)
        self.assertIs(columns.itemAt(1).widget(), dialog.fields_box)
        self.assertIs(columns.itemAt(2).widget(), dialog.discard_table.parentWidget())
        for field in dialog.fields.values():
            self.assertGreaterEqual(field.minimumHeight(), field.sizeHint().height())
        for row, spin in enumerate(dialog.discard_fields):
            self.assertGreaterEqual(dialog.discard_table.rowHeight(row), spin.sizeHint().height() + 6)

    def test_ramp_controls_and_incremental_search_are_saved_separately(self):
        dialog = self.make_dialog()
        self.assertTrue(dialog.baseline_first.isChecked())
        self.assertEqual(dialog.search_mode.currentData(), 'explore_refine')
        self.assertFalse(dialog.search_step.isHidden())
        dialog.custom.setChecked(True)
        dialog.ramp_enabled.setChecked(True)
        dialog.fields['ramp_steps'].setValue(28_000)
        dialog.search_mode.setCurrentIndex(dialog.search_mode.findData('incremental'))
        dialog.search_step.setValue(7.5)
        self.assertFalse(dialog.search_step.isHidden())
        dialog._accept()
        self.assertEqual(dialog.overrides['ramp_steps'], 28_000)
        self.assertEqual(dialog.overrides['ramp_ps'], 0.)
        self.assertEqual(dialog.overrides['production_steps'], 500000)
        self.assertEqual(dialog.search, dict(mode='incremental', baseline_first=True, step_percent=7.5,
                                            explore_trials=30, refine_trials=20))
        dialog._reset()
        self.assertFalse(dialog.ramp_enabled.isChecked())
        dialog._accept()
        self.assertEqual(dialog.overrides, {})
        self.assertEqual(dialog.search['mode'], 'incremental')

    def test_all_lengths_use_steps_and_timestep_only_changes_displayed_time(self):
        dialog = self.make_dialog()
        dialog.custom.setChecked(True)
        dialog.ramp_enabled.setChecked(True)
        for key, value in (('ramp_steps', 46_667), ('equilibration_steps', 20_000),
                           ('production_steps', 500_000)):
            self.assertIsInstance(dialog.fields[key], QtWidgets.QSpinBox)
            self.assertEqual(dialog.fields[key].suffix(), ' steps')
            dialog.fields[key].setValue(value)
        self.assertNotIn('ramp_ps', dialog.fields)
        self.assertIn('Equilibrate 20 ps → sample 500 ps', dialog.summary.text())
        dialog.fields['timestep_fs'].setValue(.5)
        self.assertIn('Equilibrate 10 ps → sample 250 ps', dialog.summary.text())
        self.assertEqual(dialog.fields['ramp_steps'].value(), 46_667)
        self.assertEqual(dialog.fields['equilibration_steps'].value(), 20_000)
        self.assertEqual(dialog.fields['production_steps'].value(), 500_000)
        dialog._accept()
        self.assertEqual(dialog.overrides['ramp_steps'], 46_667)

    def test_legacy_duration_loads_as_steps_without_rewriting_until_saved(self):
        text = 'units real\n' + SMOKE_MARKER + '\n' + '\n'.join(
            run_lines(dict(DEFAULT_PROTOCOL, ramp_ps=12.), 'id type x y z'))
        dialog = SimulationProtocolDialog({}, [], text)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.fields['ramp_steps'].value(), 28_000)
        dialog._accept()
        self.assertEqual(dialog.overrides, {})
        dialog.custom.setChecked(True)
        dialog._accept()
        self.assertEqual(dialog.overrides['ramp_steps'], 28_000)
        self.assertEqual(dialog.overrides['ramp_ps'], 0.)

    def test_saved_ramp_off_overrides_a_step_based_source(self):
        text = 'units real\n' + SMOKE_MARKER + '\n' + '\n'.join(
            run_lines(dict(DEFAULT_PROTOCOL, ramp_steps=28_000), 'id type x y z'))
        dialog = SimulationProtocolDialog(dict(DEFAULT_PROTOCOL), [], text)
        self.addCleanup(dialog.deleteLater)
        self.assertFalse(dialog.ramp_enabled.isChecked())
        self.assertEqual(dialog.fields['ramp_steps'].value(), 0)
        self.assertNotIn('Ramp ', dialog.summary.text())

    def test_separate_phase_budgets_are_compact_and_persisted(self):
        dialog = self.make_dialog()
        self.assertFalse(dialog.explore_trials.isHidden())
        self.assertFalse(dialog.refine_trials.isHidden())
        dialog.explore_trials.setValue(40)
        dialog.refine_trials.setValue(15)
        dialog._accept()
        self.assertEqual(dialog.search['explore_trials'], 40)
        self.assertEqual(dialog.search['refine_trials'], 15)
        self.assertEqual(dialog.overrides, {})
        dialog.search_mode.setCurrentIndex(dialog.search_mode.findData('current'))
        self.assertTrue(dialog.explore_trials.isHidden())
        self.assertTrue(dialog.refine_trials.isHidden())

    def test_discard_is_per_property_and_cancel_does_not_mutate_callers(self):
        dialog = self.make_dialog()
        original = deepcopy(self.properties)
        dialog.discard_fields[0].setValue(20.)
        dialog.reject()
        self.assertEqual(self.properties, original)
        dialog._accept()
        self.assertEqual(dialog.overrides, {})
        self.assertEqual(dialog.properties[0]['skip_fraction'], .2)
        self.assertEqual(self.properties, original)

    def test_source_reset_restores_simulation_and_keeps_discard(self):
        dialog = self.make_dialog()
        dialog.custom.setChecked(True)
        dialog.fields['equilibration_steps'].setValue(20_000)
        dialog.discard_fields[0].setValue(10.)
        dialog._reset()
        dialog._accept()
        self.assertEqual(dialog.overrides, {})
        self.assertEqual(dialog.fields['equilibration_steps'].value(), 0)
        self.assertEqual(dialog.properties[0]['skip_fraction'], .1)

    def test_insufficient_retained_samples_are_rejected(self):
        dialog = self.make_dialog()
        dialog.custom.setChecked(True)
        dialog.fields['production_steps'].setValue(200)
        dialog.discard_fields[0].setValue(50.)
        with patch.object(QtWidgets.QMessageBox, 'warning') as warning:
            dialog._accept()
        warning.assert_called_once()
        self.assertNotEqual(dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)

    def test_storage_defaults_and_threshold_visibility(self):
        dialog = self.make_dialog()
        self.assertEqual(dialog.storage_mode.currentData(), 'all')
        self.assertTrue(dialog.milestone_threshold.isHidden())
        self.assertTrue(dialog.storage_warning.isHidden())
        dialog.storage_mode.setCurrentIndex(dialog.storage_mode.findData('milestones'))
        self.assertFalse(dialog.milestone_threshold.isHidden())
        self.assertFalse(dialog.storage_warning.isHidden())
        dialog.milestone_threshold.setValue(7.5)
        dialog._reset()  # Source protocol reset must not reset storage.
        dialog._accept()
        self.assertEqual(dialog.overrides, {})
        self.assertEqual(dialog.trial_storage, {'mode': 'milestones', 'threshold_percent': 7.5})
        dialog.storage_mode.setCurrentIndex(dialog.storage_mode.findData('best'))
        self.assertTrue(dialog.milestone_threshold.isHidden())
        dialog._accept()
        self.assertEqual(dialog.trial_storage['mode'], 'best')

    def test_storage_load_and_cancel_do_not_mutate_caller(self):
        settings = {'mode': 'milestones', 'threshold_percent': 12.5}
        dialog = SimulationProtocolDialog({}, [], trial_storage=settings)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.storage_mode.currentData(), 'milestones')
        self.assertEqual(dialog.milestone_threshold.value(), 12.5)
        dialog.storage_mode.setCurrentIndex(dialog.storage_mode.findData('best'))
        dialog.reject()
        self.assertEqual(settings, {'mode': 'milestones', 'threshold_percent': 12.5})
        self.assertEqual(dialog.trial_storage, settings)


if __name__ == '__main__':
    unittest.main()
