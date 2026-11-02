"""Manual guide regressions. No downloads or scientific jobs are performed."""
import os
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtCore, QtWidgets

from openmscg_gui.project import Project, SystemSpec
from openmscg_gui.tutorial import (
    SQUALANE_STEPS, STEPS, StartupDialog, TutorialController,
    tutorial_check, tutorial_download_link,
)


class FakeRunner(QtCore.QObject):
    started = QtCore.Signal(str, object)
    finished = QtCore.Signal(str, object, bool)

    def __init__(self):
        super().__init__()
        self.running = False
        self.start = Mock()


class GuideWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.runner = FakeRunner()
        self.project = Project(path=Path('/unused/tutorial/project.json'), systems=[SystemSpec()])
        self.current_system_id = self.project.systems[0].id
        self.new_project = Mock()
        self.open_project = Mock()
        self.add_system = Mock()
        self._bead_dialog = None
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        for index in range(8):
            self.tabs.addTab(QtWidgets.QWidget(), str(index))
        def button(tab):
            return QtWidgets.QPushButton('Action', self.tabs.widget(tab))
        self.system_list = button(0)
        self.canvas = button(0)
        self.import_page = SimpleNamespace(inputs_button=button(0), run_button=button(0),
                                           inputs_dialog=QtWidgets.QDialog(self))
        self.mapping_page = SimpleNamespace(open_bead_editor_button=button(1), run_button=button(1))
        self.distribution_page = SimpleNamespace(interactions=button(2))
        self.force_page = SimpleNamespace(run_button=button(3))
        self.post_page = SimpleNamespace(run_button=button(4))
        self.lammps_page = SimpleNamespace(run_button=button(5))
        self.optimize_page = SimpleNamespace(protocol_button=button(6), add_property_button=button(6))
        self.validation_page = SimpleNamespace(run_button=button(7), result_tabs=button(7),
            viewer=SimpleNamespace(load_aa_properties_button=button(7)))
        self.resize(1200, 800)


class TutorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = GuideWindow()
        self.addCleanup(self.window.deleteLater)
        self.guide = TutorialController(self.window)
        self.addCleanup(self.guide.stop)
        self.window.show()
        self.app.processEvents()

    def step(self, *, check=None, stage=None):
        self.guide.index = next(index for index, item in enumerate(self.guide.steps)
                                if (check is not None and item.check == check) or
                                (stage is not None and item.stage == stage))
        self.guide.show_step()

    def test_welcome_has_three_choices_and_tutorial_does_not_need_bundled_files(self):
        dialog = StartupDialog()
        self.addCleanup(dialog.deleteLater)
        buttons = {button.text(): button for button in dialog.findChildren(QtWidgets.QPushButton)}
        self.assertEqual(set(buttons), {'New project…', 'Open project…', 'Follow the tutorial', 'Not now'})
        self.assertTrue(buttons['Follow the tutorial'].isEnabled())
        buttons['Follow the tutorial'].click()
        self.assertEqual(dialog.choice, 'tutorial')

    def test_tour_is_nonmodal_and_never_applies_settings_or_starts_actions(self):
        before = deepcopy(self.window.project)
        with patch('openmscg_gui.tutorial.QtGui.QDesktopServices.openUrl') as browser:
            self.guide.start(mode='squalane')
            self.assertFalse(self.guide.dialog.isModal())
            for index in range(len(SQUALANE_STEPS)):
                self.guide.index = index
                self.guide.show_step()
            browser.assert_not_called()
        self.assertEqual(self.window.project, before)
        self.window.new_project.assert_not_called()
        self.window.open_project.assert_not_called()
        self.window.add_system.assert_not_called()
        self.window.runner.start.assert_not_called()

    def test_first_step_available_without_project_then_waits_for_manual_creation(self):
        self.window.project = None
        self.window.current_system_id = None
        self.guide.start(mode='squalane')
        self.assertEqual(self.guide.index, 0)
        self.assertTrue(self.guide.next_button.isEnabled())
        self.guide.next()
        self.assertEqual(self.guide.steps[self.guide.index].check, 'project')
        self.assertFalse(self.guide.next_button.isEnabled())

    def test_guide_has_navigation_only_and_does_not_switch_tabs(self):
        self.window.tabs.setCurrentIndex(5)
        self.guide.start(mode='squalane')
        self.guide.next()
        self.assertEqual(self.window.tabs.currentIndex(), 5)
        buttons = {button.text() for button in self.guide.dialog.findChildren(QtWidgets.QPushButton)}
        self.assertEqual(buttons, {'Exit guide', 'Back', 'Next'})
        self.assertIn('File → New project', self.guide.steps[self.guide.index].text)

    def test_inputs_step_waits_for_inspection_then_only_requests_review(self):
        system = self.window.project.systems[0]
        with TemporaryDirectory() as directory:
            structure, trajectory = Path(directory) / 'structure.data', Path(directory) / 'aa.lammpstrj'
            structure.touch()
            trajectory.touch()
            system.structure_file, system.trajectory = str(structure), str(trajectory)
            self.guide.start(mode='squalane')
            self.step(stage='inspect')
            self.assertFalse(self.guide.next_button.isEnabled())
            self.assertIn('Click Inspect inputs in that window', self.guide.text.text())
            system.stages['inspect'] = 'complete'
            self.guide.refresh()
            self.assertTrue(self.guide.next_button.isEnabled())
            self.assertEqual(self.guide.text.text(), 'Inspection complete. Click Next.')
            self.guide.next()
            self.assertEqual(self.guide.steps[self.guide.index].title, 'Review the detected species')
            self.assertNotIn('Click Inspect inputs', self.guide.text.text())

    def test_bead_progress_observes_descriptors_not_suggested_labels(self):
        system = self.window.project.systems[0]
        system.stages['inspect'] = 'complete'
        system.detected_species = [dict(id='squalane', formula='C30H62')]
        self.guide.start(mode='squalane')
        self.step(check='bead:CH3')
        self.assertFalse(self.guide.next_button.isEnabled())
        system.custom_mapping_rules.append(dict(bead_label='B1', species_id='squalane',
                                               descriptor=dict(formula='CH3')))
        self.guide.refresh()
        self.assertTrue(self.guide.next_button.isEnabled())
        self.guide.next()
        self.assertEqual(self.guide.steps[self.guide.index].check, 'bead:CH2')
        self.assertFalse(self.guide.next_button.isEnabled())

    def test_optional_steps_can_be_skipped_without_marking_stages_complete(self):
        system = self.window.project.systems[0]
        system.stages['lammps'] = 'complete'
        before = deepcopy(self.window.project)
        self.guide.start(mode='squalane')
        self.step(stage='optimize')
        self.assertEqual(self.guide.next_button.text(), 'Skip')
        for _ in range(4):
            self.assertTrue(self.guide.next_button.isEnabled())
            self.guide.next()
        self.assertFalse(self.guide.active)
        self.assertEqual(self.window.project, before)
        self.assertEqual(system.stages['validate'], 'pending')

    def test_switching_project_resets_instructions_but_does_not_reset_project_data(self):
        self.guide.start(mode='squalane')
        self.guide.index = 10
        other = SystemSpec()
        self.window.project.systems.append(other)
        self.window.current_system_id = other.id
        self.guide.refresh()
        self.assertEqual(self.guide.steps[self.guide.index].check, 'project')
        self.assertEqual(len(self.window.project.systems), 2)

    def test_coach_stays_owned_by_main_window_when_editor_finishes(self):
        self.window.project.systems[0].stages['inspect'] = 'complete'
        self.guide.start(mode='squalane')
        self.step(check='bead:CH3')
        editor = QtWidgets.QDialog(self.window)
        self.addCleanup(editor.deleteLater)
        editor.canvas = QtWidgets.QWidget(editor)
        self.window._bead_dialog = editor
        editor.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        editor.show()
        self.app.processEvents()
        self.guide.refresh()
        self.assertIs(self.guide.dialog.parentWidget(), self.window)
        self.assertFalse(self.guide.dialog.isModal())
        self.assertFalse(self.guide.next_button.isEnabled())
        editor.accept()
        self.app.processEvents()
        self.guide.refresh()
        self.assertIs(self.guide.dialog.parentWidget(), self.window)
        self.assertTrue(self.guide.active)
        self.assertTrue(self.guide.dialog.isVisible())

    def test_closing_or_destroying_an_editor_cannot_close_the_guide(self):
        self.window.project.systems[0].stages['inspect'] = 'complete'
        self.guide.start(mode='squalane')
        self.step(check='bead:CH3')
        for action in ('reject', 'close', 'deleteLater'):
            with self.subTest(action=action):
                editor = QtWidgets.QDialog(self.window)
                self.window._bead_dialog = editor
                editor.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
                editor.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
                editor.show()
                self.app.processEvents()
                self.guide.refresh()
                self.assertIs(self.guide.dialog.parentWidget(), self.window)
                getattr(editor, action)()
                QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
                self.app.processEvents()
                self.window._bead_dialog = None
                self.guide.refresh()
                self.assertTrue(self.guide.active)
                self.assertTrue(self.guide.dialog.isVisible())
                self.assertEqual(self.guide.steps[self.guide.index].check, 'bead:CH3')

    def test_closing_inputs_or_a_file_picker_restores_the_guide(self):
        self.guide.start(mode='squalane')
        self.step(stage='inspect')
        inputs = self.window.import_page.inputs_dialog
        inputs.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        inputs.open()
        self.app.processEvents()
        self.guide.refresh()
        self.assertIs(self.guide.dialog.parentWidget(), self.window)
        self.assertFalse(self.guide.next_button.isEnabled())
        picker = QtWidgets.QDialog(inputs)
        picker.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        picker.show()
        self.app.processEvents()
        self.guide.refresh()
        self.assertFalse(self.guide.dialog.isVisible())
        picker.reject()
        inputs.reject()
        self.app.processEvents()
        self.guide.refresh()
        self.assertTrue(self.guide.active)
        self.assertTrue(self.guide.dialog.isVisible())

    def test_general_introduction_is_still_available(self):
        self.guide.start(mode='squalane')
        self.guide.start()
        self.assertIs(self.guide.steps, STEPS)
        self.guide.index = 0
        self.guide.show_step()
        self.assertIn('File → New project', STEPS[0].text)

    def test_general_introduction_does_not_repeat_completed_inspection(self):
        self.window.project.systems[0].stages['inspect'] = 'complete'
        self.guide.start()
        self.assertIn('Inspection complete.', self.guide.text.text())
        self.assertNotIn('Click Inspect inputs', self.guide.text.text())
        self.assertTrue(self.guide.next_button.isEnabled())

    def test_running_jobs_hide_the_tutorial_without_mutating_states(self):
        self.guide.start(mode='squalane')
        before = deepcopy(self.window.project)
        self.window.runner.running = True
        self.guide.refresh()
        self.assertFalse(self.guide.dialog.isVisible())
        self.assertEqual(self.window.project, before)
        self.window.runner.running = False
        self.guide.refresh()
        self.assertTrue(self.guide.dialog.isVisible())


class TutorialResourceTests(unittest.TestCase):
    def test_instructions_are_short_and_inspection_is_a_single_action_step(self):
        for step in STEPS + SQUALANE_STEPS:
            self.assertLessEqual(len(step.text.split()), 50, step.title)
        self.assertEqual(sum(step.stage == 'inspect' for step in SQUALANE_STEPS), 1)

    def test_url_placeholder_invalid_schemes_and_configured_download(self):
        with TemporaryDirectory() as directory, patch('openmscg_gui.tutorial.TUTORIAL_DATA', Path(directory)):
            path = Path(directory) / 'example.json'
            self.assertEqual(tutorial_download_link(), ('<LINK>', None))
            for url in ('<LINK>', 'file:///tmp/data.zip', 'javascript:alert(1)',
                        'https://user:secret@example.org/data.zip', 'https://example.org/bad path'):
                path.write_text(json.dumps(dict(download_url=url)))
                self.assertIsNone(tutorial_download_link()[1])
            url = 'https://example.org/squalane-inputs.zip'
            path.write_text(json.dumps(dict(download_url=url)))
            self.assertEqual(tutorial_download_link(), (url, url))
            path.write_text('[]')
            self.assertEqual(tutorial_download_link(), ('<LINK>', None))

    def test_state_checks_read_inputs_and_matching_rules_only(self):
        system = SystemSpec()
        system.detected_species = [dict(id='squalane', formula='C30H62')]
        system.custom_mapping_rules = [dict(species_id='other', descriptor=dict(formula='CH3'))]
        self.assertFalse(tutorial_check('bead:CH3', system))
        system.custom_mapping_rules[0]['species_id'] = 'squalane'
        self.assertTrue(tutorial_check('bead:CH3', system))
        self.assertFalse(tutorial_check('inputs', system))
        with TemporaryDirectory() as directory:
            structure = Path(directory) / 'structure.data'
            trajectory = Path(directory) / 'aa.lammpstrj'
            structure.touch()
            trajectory.touch()
            system.structure_file, system.trajectory = str(structure), str(trajectory)
            before = deepcopy(system)
            self.assertTrue(tutorial_check('inputs', system))
            self.assertEqual(system, before)


if __name__ == '__main__':
    unittest.main()
