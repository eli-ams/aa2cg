"""Linked comparison controls. Run manually with the GUI dependencies."""
import os
from types import SimpleNamespace
import unittest

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets
from openmscg_gui.trajectory_comparison import TrajectoryComparison, relative_frame


class RelativeFrameTests(unittest.TestCase):
    def test_unequal_lengths_endpoints_midpoint_and_single_frame(self):
        for count in (0, 1, 5, 11, 101):
            self.assertEqual(relative_frame(0, 100, count), 0)
            self.assertEqual(relative_frame(100, 100, count), max(0, count - 1))
            self.assertEqual(relative_frame(50, 100, count), max(0, (count - 1) // 2))
        self.assertEqual(relative_frame(9, 0, 20), 0)
        self.assertEqual(relative_frame(-10, 100, 11), 0)
        self.assertEqual(relative_frame(120, 100, 11), 10)


class TrajectoryComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.pair = TrajectoryComparison()
        self.addCleanup(self.pair.deleteLater)
        self.addCleanup(self.pair.shutdown)
        # No source, rendering jobs or OpenGL context are needed for these controls.
        self.pair._syncing = True
        for index, view in enumerate(self.pair.views):
            species = [SimpleNamespace(identity='a', molecules=[[], [], []], molecule_ids=[1, 2, 3]),
                       SimpleNamespace(identity='b', molecules=[[], []], molecule_ids=[4, 5])]
            if index:
                species.reverse()  # Match stable identities, not combo indices.
            view._model = SimpleNamespace(species=species)
            view._reader = SimpleNamespace(frames=list(range(11 if index == 0 else 5)))
            view.species.blockSignals(True)
            for item in species:
                view.species.addItem(item.identity, item.identity)
            view.species.blockSignals(False)
            view._species_changed()
            view.slider.setRange(0, len(view._reader.frames) - 1)
        self.pair._syncing = False
        self.pair._state_changed()
        self.settle()

    def settle(self):
        for view in self.pair.views:
            view._debounce.stop()
            view._dirty = view._busy = False

    def test_one_bar_and_shared_progress(self):
        self.assertTrue(all(v.playback_bar.isHidden() for v in self.pair.views))
        self.pair.slider.setValue(5)
        self.assertEqual([v.slider.value() for v in self.pair.views], [5, 2])
        self.pair.slider.setValue(10)
        self.assertEqual([v.slider.value() for v in self.pair.views], [10, 4])
        self.assertFalse(any(v._timer.isActive() for v in self.pair.views))

    def test_shared_selection_in_both_directions(self):
        aa, cg = self.pair.views
        aa.instance.setValue(3)
        self.assertEqual(cg.species.currentData(), 'a')
        self.assertEqual(cg.instance.value(), 3)
        cg.species.setCurrentIndex(cg.species.findData('b'))
        cg.instance.setValue(2)
        self.assertEqual(aa.species.currentData(), 'b')
        self.assertEqual(aa.instance.value(), 2)
        aa.scope.setCurrentIndex(0)
        self.assertEqual(cg.scope.currentIndex(), 0)

    def test_display_preferences_apply_to_both_and_isolation_restores_them(self):
        aa, cg = self.pair.views
        aa.scope.setCurrentIndex(0)
        aa._set_pbc_display(False)
        aa._set_box_display(True)
        self.assertEqual(aa._display_options(), cg._display_options())
        aa.scope.setCurrentIndex(2)
        self.assertTrue(cg._display_options()['pbc'])
        aa.scope.setCurrentIndex(0)
        self.assertFalse(cg._display_options()['pbc'])

    def test_playback_waits_for_both_and_pauses_on_missing_or_failed_source(self):
        self.pair._toggle_play()
        self.assertTrue(self.pair._timer.isActive())
        self.pair.cg_view._busy = True
        self.pair._advance()
        self.assertEqual(self.pair.slider.value(), 0)
        self.settle()
        self.pair._advance()
        self.assertEqual(self.pair.slider.value(), 1)
        self.pair.cg_view._preview_error = True
        self.pair._state_changed()
        self.assertFalse(self.pair._timer.isActive())
        self.assertFalse(self.pair.play.isEnabled())
        self.pair.cg_view._reader = None
        self.pair._state_changed()
        self.assertFalse(self.pair.slider.isEnabled())

    def test_single_frames_disable_play(self):
        for view in self.pair.views:
            view._reader.frames = [0]
        self.pair._state_changed()
        self.assertEqual(self.pair.slider.maximum(), 0)
        self.assertFalse(self.pair.play.isEnabled())

    def test_camera_links_bidirectionally_without_feedback(self):
        aa, cg = [v.renderer for v in self.pair.views]
        self.assertIsNotNone(aa)
        self.assertIsNotNone(cg)
        for renderer in (aa, cg):
            renderer.yaw = .7
            renderer.pitch = -.2
            renderer.distance = 12.
            renderer.target = renderer.scene.centre + np.array([1., 2., 3.])
            renderer.cameraChanged.emit(renderer.camera_state())
            for peer in (aa, cg):
                self.assertEqual(peer.yaw, .7)
                self.assertEqual(peer.pitch, -.2)
                self.assertEqual(peer.distance, 12.)
                np.testing.assert_allclose(peer.target - peer.scene.centre, [1., 2., 3.])
        self.pair._scene_changed(self.pair.cg_view, True)
        self.assertEqual(cg.distance, 12.)  # Late scene updates preserve linked interaction.

    def test_frame_view_uses_one_scale_fitting_the_larger_scene(self):
        aa, cg = [v.renderer for v in self.pair.views]
        aa.fit_distance, cg.fit_distance = 8., 15.
        self.pair._refit = True
        self.pair._scene_changed(self.pair.cg_view, True)
        self.assertEqual(aa.distance, 15.)
        self.assertEqual(cg.distance, 15.)

    def test_shutdown_stops_shared_playback(self):
        self.pair._toggle_play()
        self.assertTrue(self.pair._timer.isActive())
        self.pair.shutdown()
        self.assertFalse(self.pair._timer.isActive())


if __name__ == '__main__':
    unittest.main()
