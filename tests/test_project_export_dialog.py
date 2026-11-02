"""Compact export controls; these cases are for manual GUI test runs."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets

from openmscg_gui.project import Project
from openmscg_gui.project_export import plan_export
from openmscg_gui.project_export_dialog import ProjectExportDialog


class ProjectExportDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Project.create(Path(self.temporary.name) / 'project')
        # Exercise dialog state transitions without starting background work.
        with patch('openmscg_gui.project_export_dialog.QtCore.QTimer.singleShot'):
            self.dialog = ProjectExportDialog(self.project)
        self.addCleanup(self.dialog.deleteLater)

    def worker(self, value=None, problem='', cancelled=False):
        return SimpleNamespace(value=value, problem=problem, was_cancelled=cancelled,
                               cancel=Event(), deleteLater=Mock())

    def test_default_is_best_and_size_is_explicitly_before_compression(self):
        self.assertEqual(self.dialog.preset.count(), 3)
        self.assertEqual(self.dialog.preset.currentData(), 'best')
        self.assertFalse(self.dialog.export_button.isEnabled())
        plan = plan_export(self.project)
        self.dialog.job = 'plan'
        self.dialog.worker = self.worker(value=plan)
        self.dialog._finished()
        self.assertTrue(self.dialog.export_button.isEnabled())
        self.assertIn('before compression', self.dialog.estimate.text())

    def test_cancel_waits_for_worker_cleanup_before_closing(self):
        worker = self.worker(cancelled=True)
        self.dialog.worker = worker
        self.dialog.reject()
        self.assertTrue(worker.cancel.is_set())
        self.assertTrue(self.dialog.busy)
        self.assertFalse(self.dialog.cancel_button.isEnabled())
        self.dialog._finished()
        self.assertFalse(self.dialog.busy)
        worker.deleteLater.assert_called_once()

    def test_write_error_leaves_the_dialog_ready_for_a_new_destination(self):
        self.dialog.plan = plan_export(self.project)
        self.dialog.job = 'write'
        self.dialog.worker = self.worker(problem='Disk full')
        with patch.object(QtWidgets.QMessageBox, 'warning') as warning:
            self.dialog._finished()
        warning.assert_called_once()
        self.assertTrue(self.dialog.export_button.isEnabled())
        self.assertTrue(self.dialog.preset.isEnabled())
        self.assertIn('unchanged', self.dialog.status.text())


if __name__ == '__main__':
    unittest.main()
