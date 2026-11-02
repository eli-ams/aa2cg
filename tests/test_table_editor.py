"""GUI recipe/persistence cases, for manual execution in the GUI environment."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets
from openmscg_gui.table_editor import TableProcessingExplorer


class TableEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tables = self.root / 'output' / 'tables'
        tables.mkdir(parents=True)
        for name in ('Bond_A-B.table', 'Bond_B-B.table'):
            (tables / name).write_text('# test\n\nBond\nN 161\n\n' + ''.join(
                '{} {} {} {}\n'.format(i + 1, 1 + i / 80., (i / 80. - 1) ** 2, -2 * (i / 80. - 1))
                for i in range(161)), encoding='utf-8')
        self.editor = TableProcessingExplorer()
        self.addCleanup(self.editor.deleteLater)
        self.addCleanup(self.editor._timer.stop)
        self.changes = []
        self.editor.overrideChanged.connect(lambda name, value: self.changes.append((name, value)))
        self.editor.load_system_root(self.root)
        self.editor._timer.stop()  # These cases inspect controls without starting preview jobs.

    def test_custom_controls_save_per_interaction_and_reset(self):
        self.assertEqual(self.changes, [])
        self.editor.mode.setCurrentIndex(1)
        self.editor.smooth.setValue(7)
        self.editor._commit()
        self.assertEqual(self.changes[-1][0], 'Bond_A-B.table')
        self.assertEqual(self.changes[-1][1]['smooth'], 7)
        self.editor.table_list.setCurrentRow(1)
        self.assertEqual(self.editor.mode.currentIndex(), 0)
        self.editor.table_list.setCurrentRow(0)
        self.assertEqual(self.editor.smooth.value(), 7)
        self.editor._reset()
        self.assertEqual(self.changes[-1], ('Bond_A-B.table', None))
        self.assertEqual(self.editor.mode.currentIndex(), 0)
        self.assertEqual(self.editor.smooth.value(), 5)

    def test_invalid_window_restores_saved_recipe_and_old_jobs_are_ignored(self):
        self.editor.mode.setCurrentIndex(1)
        before = list(self.changes)
        self.editor.smooth.setValue(6)
        self.editor._commit()
        self.assertEqual(self.changes, before)
        self.assertEqual(self.editor.smooth.value(), 5)
        detail = self.editor.detail.text()
        self.editor._preview_finished(self.editor._token - 1, None, 'stale error')
        self.assertEqual(self.editor.detail.text(), detail)


if __name__ == '__main__':
    unittest.main()
