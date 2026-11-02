"""Shared-view menu regression cases; run manually with GUI dependencies."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets
from openmscg_gui.trajectory_view import TrajectoryView


class TrajectoryDisplaySettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_view(self, **kwargs):
        view = TrajectoryView(**kwargs)
        self.addCleanup(view.deleteLater)
        self.addCleanup(view.shutdown)
        view._model = SimpleNamespace(scene=SimpleNamespace(cell_matrix=np.eye(3)), species=[])
        return view

    def menu_state(self, view):
        state = {}
        def inspect(menu, *_):
            for action in menu.actions():
                state[action.text()] = (action.isEnabled(), action.isChecked())
        with patch.object(QtWidgets.QMenu, 'exec', inspect):
            view._show_view_menu(view)
        return state

    def test_shared_menu_in_inspect_mapping_and_validation_variants(self):
        for options in ({}, {'mapping': True}, {'cg': True}, {'compact_controls': False}):
            with self.subTest(options=options):
                view = self.make_view(**options)
                initial = self.menu_state(view)
                self.assertEqual(initial['Show simulation box'], (True, True))
                self.assertEqual(initial['Make molecules whole (PBC)'], (True, True))
                view._set_pbc_display(False)
                view.scope.setCurrentIndex(2)
                isolated = self.menu_state(view)
                self.assertEqual(isolated['Make molecules whole (PBC)'], (False, True))
                self.assertEqual(isolated['Show simulation box'], (True, False))
                view._set_pbc_display(True)  # A disabled control cannot overwrite the preference.
                view._set_box_display(True)
                view.follow.setChecked(False)
                self.assertTrue(view._display_options()['pbc'])
                view.scope.setCurrentIndex(0)
                self.assertEqual(view._display_options(), {'pbc': False, 'show_box': True})
                view._set_box_display(False)
                view.scope.setCurrentIndex(2)
                self.assertTrue(view._display_options()['show_box'])
                view.scope.setCurrentIndex(0)
                self.assertEqual(view._display_options(), {'pbc': False, 'show_box': False})

    def test_missing_cell_disables_both_controls(self):
        view = self.make_view()
        view._model.scene.cell_matrix = None
        state = self.menu_state(view)
        self.assertFalse(state['Show simulation box'][0])
        self.assertFalse(state['Make molecules whole (PBC)'][0])


if __name__ == '__main__':
    unittest.main()
