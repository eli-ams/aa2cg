"""Welcome screen and optional, modeless guide through the real workflow."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from urllib.parse import urlsplit

from PySide6 import QtCore, QtGui, QtWidgets

from . import APP_TITLE

TUTORIAL_DATA = Path(__file__).with_name('tutorial_data')


def onboarding_settings():
    # User preferences belong outside the source tree and scientific projects.
    # Keep the legacy settings namespace so existing preferences survive renaming.
    return QtCore.QSettings('OpenMSCGWorkbench', 'Onboarding')


def tutorial_download_link():
    """Return display text and a usable URL, never download or inspect inputs."""
    try:
        manifest = json.loads((TUTORIAL_DATA / 'example.json').read_text(encoding='utf-8'))
        if not isinstance(manifest, dict):
            return '<LINK>', None
        text = manifest.get('download_url', '<LINK>')
        if not isinstance(text, str) or not text.strip():
            return '<LINK>', None
        text = text.strip()
        url = urlsplit(text)
        usable = (url.scheme in ('https', 'http') and bool(url.hostname)
                  and not url.username and not url.password and not any(c.isspace() for c in text)
                  and '<' not in text and '>' not in text)
        return text, text if usable else None
    except (OSError, ValueError, TypeError):
        return '<LINK>', None


class StartupDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.choice = ''
        self.setWindowTitle('Welcome to ' + APP_TITLE)
        self.resize(760, 265)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(12)
        title = QtWidgets.QLabel('Create a coarse-grained force field')
        title.setObjectName('pageTitle')
        description = QtWidgets.QLabel('Start from your LAMMPS structure and trajectory, or continue an existing project.')
        description.setWordWrap(True)
        description.setObjectName('pageDescription')
        layout.addWidget(title)
        layout.addWidget(description)
        choices = QtWidgets.QHBoxLayout()
        for label, choice in (('New project…', 'new'), ('Open project…', 'open'),
                              ('Follow the tutorial', 'tutorial')):
            button = QtWidgets.QPushButton(label)
            button.setMinimumHeight(56)
            if choice == 'tutorial':
                button.setObjectName('primaryButton')
                button.setToolTip('A floating Squalane guide. Download the raw inputs and perform each action yourself; defaults stay unchanged.')
            button.clicked.connect(lambda checked=False, choice=choice: self._choose(choice))
            choices.addWidget(button)
        layout.addLayout(choices)

        inputs = QtWidgets.QLabel('Inputs: a structure with masses and bonds, and a matching trajectory with atomic forces (fx, fy, fz).')
        inputs.setWordWrap(True)
        inputs.setObjectName('note')
        layout.addWidget(inputs)
        note = QtWidgets.QLabel('The Squalane tutorial starts from raw inputs. You create the project, mapping and results yourself.')
        note.setObjectName('note')
        note.setWordWrap(True)
        layout.addWidget(note)
        footer = QtWidgets.QHBoxLayout()
        self.show_on_startup = QtWidgets.QCheckBox('Show welcome on startup')
        self.show_on_startup.setChecked(onboarding_settings().value('show_welcome', True, type=bool))
        later = QtWidgets.QPushButton('Not now')
        later.setFlat(True)
        later.clicked.connect(self.reject)
        footer.addWidget(self.show_on_startup)
        footer.addStretch()
        footer.addWidget(later)
        layout.addLayout(footer)

    def _choose(self, choice):
        self.choice = choice
        self.accept()


@dataclass
class GuideStep:
    title: str
    text: str
    tab: int
    stage: str | None
    target: str
    prerequisite: str | None = None
    optional: bool = False
    check: str | None = None
    done_text: str = 'Done. Click Next.'


STEPS = (
    GuideStep('Choose a project',
        'Choose File → New project… or File → Open project….\n'
        'Select a system in the Systems list. Use Add system if the list is empty.',
        0, None, 'system_list', check='project', done_text='Project ready. Click Next.'),
    GuideStep('Load and inspect inputs',
        'Open 1 · Inspect → Load / change inputs….\n'
        'Choose your structure and force-containing trajectory.\n'
        'Click Inspect inputs in that window.',
        0, 'inspect', 'import_page.inputs_button',
        done_text='Inspection complete.\nOpen Details / checks… and review the results. Then click Next.'),
    GuideStep('Define the bead mapping',
        'Open 2 · Map and select a species.\n'
        'Use Create bead rule… to group atoms into beads.\n'
        'Click Build mapped trajectory.',
        1, 'cgmodel', 'mapping_page.open_bead_editor_button', prerequisite='inspect'),
    GuideStep('Review the fitting plan',
        'Open 3 · Configure fit.\nSelect interactions to view their distributions.\nKeep the default settings.',
        2, None, 'distribution_page.interactions', prerequisite='cgmodel'),
    GuideStep('Fit the interactions',
        'Open 4 · Fit.\nKeep Independent fitting selected.\nClick Fit force field and wait.',
        3, 'force', 'force_page.run_button', prerequisite='cgmodel',
        done_text='Fitting complete.\nReview the raw curves, then click Next.'),
    GuideStep('Prepare the tables',
        'Open 5 · Prepare tables.\nKeep the default processing settings.\nClick Prepare tables and wait.',
        4, 'postprocess', 'post_page.run_button', prerequisite='force',
        done_text='Tables prepared.\nCompare the raw and processed curves, then click Next.'),
    GuideStep('Generate the LAMMPS setup',
        'Open 6 · LAMMPS.\nClick Generate LAMMPS setup and wait.',
        5, 'lammps', 'lammps_page.run_button', prerequisite='postprocess',
        done_text='LAMMPS setup ready.\nReview input.data and structure.data, then click Next.'),
    GuideStep('Refine · optional',
        'Open 7 · Refine.\nAdd property objectives and check your LAMMPS command.\n'
        'Click Optimize force field, or click Skip in this guide.',
        6, 'optimize', 'optimize_page.protocol_button', prerequisite='lammps', optional=True),
    GuideStep('Compare AA and CG',
        'Open 8 · Validate.\nClick Compare AA and CG, then review Properties, Plots and 3D visuals.\n'
        'A CG simulation trajectory is required.',
        7, 'validate', 'validation_page.run_button', prerequisite='lammps',
        done_text='Comparison ready.\nReview Properties, Plots and 3D visuals, then click Finish.'),
)


SQUALANE_STEPS = (
    GuideStep('Download the Squalane inputs',
        'Download the input ZIP from the link below.\nExtract it to a folder on your computer.\n'
        'You need structure.data, aa.lammpstrj and properties.txt. Keep default settings throughout this tutorial.',
        0, None, '', check='download'),
    GuideStep('Create a blank project',
        'In the main window, choose File → New project….\nChoose a new, empty project folder.',
        0, None, '', check='project', done_text='Project ready. Click Next.'),
    GuideStep('Load and inspect inputs',
        'Open 1 · Inspect → Load / change inputs….\n'
        'Choose structure.data and aa.lammpstrj from the extracted folder.\n'
        'Click Inspect inputs in that window and wait.',
        0, 'inspect', 'import_page.inputs_button', done_text='Inspection complete. Click Next.'),
    GuideStep('Review the detected species',
        'Open Details / checks… and review the checks.\n'
        'Confirm C30H62: 100 molecules, 92 atoms each.\nClose the checks window.',
        0, None, 'import_page.details_button', prerequisite='inspect'),
    GuideStep('Explore one molecule',
        'In the 3D panel, choose View → Show → Single molecule.\n'
        'Press Play to view its motion.\nCompare it with the 2D drawing.',
        0, None, 'canvas', prerequisite='inspect'),
    GuideStep('Map CH3 groups',
        'Open 2 · Map → Create bead rule….\n'
        'Select one terminal carbon and its 3 hydrogens (4 atoms).\n'
        'Keep the suggested label. Click Create bead rule.',
        1, None, 'mapping_page.open_bead_editor_button', prerequisite='inspect', check='bead:CH3',
        done_text='CH3 rule saved. Click Next.'),
    GuideStep('Map CH2 groups',
        'Click Create bead rule….\nSelect one unmapped carbon and its 2 hydrogens (3 atoms).\n'
        'Keep the suggested label. Click Create bead rule.',
        1, None, 'mapping_page.open_bead_editor_button', prerequisite='inspect', check='bead:CH2',
        done_text='CH2 rule saved. Click Next.'),
    GuideStep('Map CH groups',
        'Click Create bead rule….\nSelect one branch-point carbon and its hydrogen (2 atoms).\n'
        'Keep the suggested label. Click Create bead rule.',
        1, None, 'mapping_page.open_bead_editor_button', prerequisite='inspect', check='bead:CH',
        done_text='CH rule saved. Click Next.'),
    GuideStep('Build the mapped trajectory',
        'Check that every atom is mapped: 30 beads per molecule.\n'
        'Click Build mapped trajectory and wait.',
        1, 'cgmodel', 'mapping_page.run_button', prerequisite='inspect',
        done_text='Mapped trajectory ready. Click Next.'),
    GuideStep('Review Configure fit',
        'Open 3 · Configure fit.\nSelect pair, bond and angle interactions to view their distributions.\n'
        'Keep the default bounds, spacing and order.',
        2, None, 'distribution_page.interactions', prerequisite='cgmodel'),
    GuideStep('Fit the force field',
        'Open 4 · Fit.\nKeep Independent fitting selected.\nClick Fit force field and wait.',
        3, 'force', 'force_page.run_button', prerequisite='cgmodel',
        done_text='Fitting complete.\nReview the raw curves, then click Next.'),
    GuideStep('Prepare the tables',
        'Open 5 · Prepare tables.\nKeep the default processing settings.\nClick Prepare tables and wait.',
        4, 'postprocess', 'post_page.run_button', prerequisite='force',
        done_text='Tables prepared.\nCompare the raw and processed curves, then click Next.'),
    GuideStep('Generate the LAMMPS setup',
        'Open 6 · LAMMPS.\nClick Generate LAMMPS setup and wait.',
        5, 'lammps', 'lammps_page.run_button', prerequisite='postprocess',
        done_text='LAMMPS setup ready.\nReview input.data and structure.data, then click Next.'),
    GuideStep('Optional: refine with simulations',
        'Open 7 · Refine.\nChoose Density in the property menu, then click Add property.\n'
        'Check your LAMMPS command, then click Optimize force field.\nOr click Skip in this guide.',
        6, 'optimize', 'optimize_page.add_property_button', prerequisite='lammps', optional=True),
    GuideStep('Compare AA and CG distributions',
        'Open 8 · Validate.\nClick Compare AA and CG, then open Plots.\n'
        'If you have no CG simulation trajectory, click Skip.',
        7, 'validate', 'validation_page.run_button', prerequisite='lammps', optional=True,
        done_text='Comparison ready.\nOpen Plots and compare the AA and CG distributions.'),
    GuideStep('Load AA properties',
        'Open Properties → Load AA properties.\nChoose properties.txt from the extracted folder.\n'
        'Compare the AA and CG values. If no CG results exist, click Skip.',
        7, None, 'validation_page.viewer.load_aa_properties_button', prerequisite='lammps',
        optional=True, check='properties', done_text='AA properties loaded.\nCompare the AA and CG values.'),
    GuideStep('Inspect the result and save',
        'Open 3D visuals. Use View to select one molecule and press Play.\n'
        'Compare AA and CG side by side (requires a CG trajectory).\n'
        'Choose File → Save. Click Finish.',
        7, None, 'validation_page.result_tabs', prerequisite='lammps', optional=True),
)


def tutorial_check(check, system):
    """Read saved state only. Never select atoms, apply settings or start jobs."""
    if check == 'download':
        return True
    if system is None:
        return False
    if check == 'project':
        return True
    if check == 'inputs':
        return bool(system.structure_file and system.trajectory and
                    Path(system.structure_file).is_file() and Path(system.trajectory).is_file())
    if check == 'properties':
        return bool(system.aa_properties_file and Path(system.aa_properties_file).is_file())
    if check and check.startswith('bead:'):
        formula = check.partition(':')[2]
        species_ids = {item.get('id') for item in system.detected_species if item.get('formula') == 'C30H62'}
        return any(rule.get('species_id') in species_ids and
                   (rule.get('descriptor') or {}).get('formula') == formula
                   for rule in system.custom_mapping_rules)
    return False


class TutorialController(QtCore.QObject):
    """Observe workflow state; only the user's existing run buttons start jobs."""
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.active = False
        self.index = 0
        self.highlighted = None
        self.context = None
        self.mode = 'introduction'
        self.steps = STEPS
        self._editor_overlay = False
        self.dialog = QtWidgets.QDialog(window)
        self.dialog.setWindowTitle('Guided introduction')
        self.dialog.setWindowFlags(QtCore.Qt.WindowType.Tool |
                                   QtCore.Qt.WindowType.WindowTitleHint |
                                   QtCore.Qt.WindowType.WindowCloseButtonHint)
        self.dialog.setModal(False)
        self.dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.dialog.resize(520, 250)
        self.dialog.setMinimumWidth(450)
        layout = QtWidgets.QVBoxLayout(self.dialog)
        layout.setContentsMargins(16, 14, 16, 12)
        self.progress = QtWidgets.QLabel()
        self.progress.setObjectName('pageDescription')
        self.title = QtWidgets.QLabel()
        self.title.setObjectName('sectionTitle')
        self.title.setWordWrap(True)
        self.text = QtWidgets.QLabel()
        self.text.setWordWrap(True)
        self.context_label = QtWidgets.QLabel()
        self.context_label.setObjectName('pageDescription')
        self.context_label.setWordWrap(True)
        self.note = QtWidgets.QLabel()
        self.note.setObjectName('note')
        self.note.setWordWrap(True)
        for widget in (self.progress, self.title, self.text, self.context_label, self.note):
            widget.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            layout.addWidget(widget)
        self.download_row = QtWidgets.QWidget()
        download_layout = QtWidgets.QHBoxLayout(self.download_row)
        download_layout.setContentsMargins(0, 0, 0, 0)
        self.download_link = QtWidgets.QLabel()
        self.download_link.setOpenExternalLinks(False)
        self.download_link.linkActivated.connect(self._open_download)
        download_layout.addWidget(self.download_link, 1)
        layout.addWidget(self.download_row)
        self.download_row.hide()
        controls = QtWidgets.QHBoxLayout()
        exit_button = QtWidgets.QPushButton('Exit guide')
        exit_button.setFlat(True)
        self.back_button = QtWidgets.QPushButton('Back')
        self.next_button = QtWidgets.QPushButton('Next')
        self.next_button.setObjectName('primaryButton')
        for widget in (exit_button, self.back_button, self.next_button):
            controls.addWidget(widget)
        layout.addLayout(controls)
        exit_button.clicked.connect(self.stop)
        self.dialog.rejected.connect(self.stop)
        self.back_button.clicked.connect(self.back)
        self.next_button.clicked.connect(self.next)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.refresh)
        window.tabs.currentChanged.connect(self.refresh)
        window.runner.started.connect(self.refresh)
        window.runner.finished.connect(self.refresh)

    def _system(self):
        if self.window.project is not None and self.window.current_system_id:
            try:
                return self.window.project.system(self.window.current_system_id)
            except KeyError:
                pass
        return None

    def _context(self):
        # Inspection reloads the Project object; its file and system ID stay stable.
        project = self.window.project
        return (str(project.path) if project is not None else None,
                self.window.current_system_id)

    def start(self, mode='introduction'):
        if mode not in ('introduction', 'squalane'):
            raise ValueError('Unknown tutorial mode.')
        if self.active and mode == self.mode:
            self.show_step()
            return
        self.stop()
        self.mode = mode
        self.steps = SQUALANE_STEPS if mode == 'squalane' else STEPS
        self.dialog.setWindowTitle('Squalane tutorial · manual steps' if mode == 'squalane' else 'Guided introduction')
        self.active = True
        self.index = 0 if mode == 'squalane' else 1 if self._system() is not None else 0
        self.context = self._context()
        _, url = tutorial_download_link()
        self.download_link.setTextFormat(QtCore.Qt.TextFormat.RichText if url else QtCore.Qt.TextFormat.PlainText)
        self.download_link.setText('<a href="{}">Download Squalane inputs</a>'.format(escape(url, quote=True))
                                   if url else 'Download link: <LINK> (not available yet)')
        self.download_link.setToolTip(url or '')
        self.timer.start()
        self.show_step()
        self.dialog.adjustSize()
        frame = self.window.frameGeometry()
        screen = self.window.screen()
        if screen is not None:
            area = screen.availableGeometry()
            x = min(frame.right() - self.dialog.width() - 16, area.right() - self.dialog.width())
            y = min(frame.top() + 110, area.bottom() - self.dialog.height())
            self.dialog.move(max(area.left(), x), max(area.top(), y))

    def stop(self):
        self.active = False
        self.timer.stop()
        self._highlight(None)
        self.dialog.hide()
        self._set_editor_overlay(False)

    def _open_download(self, *_):
        _, url = tutorial_download_link()
        if url and not QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)):
            QtWidgets.QMessageBox.information(self.dialog, 'Open the link manually', url)

    def _set_editor_overlay(self, enabled):
        # Never parent the coach to a temporary editor: Qt may close child
        # dialogs before emitting that editor's finished signal.
        if enabled != self._editor_overlay:
            position = self.dialog.pos()
            self.dialog.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, enabled)
            self.dialog.move(position)
            self._editor_overlay = enabled

    def back(self):
        if self.index > 0:
            self.index -= 1
            self.show_step()

    def next(self):
        self.refresh()
        if not self.next_button.isEnabled():
            return
        if self.index == len(self.steps) - 1:
            self.stop()
            self.window.statusBar().showMessage('Tutorial finished.', 5000)
        else:
            self.index += 1
            self.show_step()

    def show_step(self):
        if not self.active or self.window.runner.running:
            return
        self.refresh()
        if self.dialog.isVisible():
            self.dialog.raise_()

    def refresh(self, *_):
        if not self.active:
            return
        # Remain readable alongside manual input/bead editing. Other modal
        # dialogs (file pickers, errors, job monitors) must remain unobscured.
        modal = QtWidgets.QApplication.activeModalWidget()
        instruction_modal = modal is not None and modal in (
            getattr(self.window, '_bead_dialog', None),
            getattr(getattr(self.window, 'import_page', None), 'inputs_dialog', None))
        if (self.window.runner.running or not self.window.isVisible() or self.window.isMinimized()
                or (modal is not None and modal is not self.dialog and not instruction_modal)):
            self._highlight(None)
            self._set_editor_overlay(False)
            self.dialog.hide()
            return
        self._set_editor_overlay(instruction_modal)
        context = self._context()
        if context != self.context:
            self.context = context
            self.index = 1 if self.mode == 'squalane' else 0
        system = self._system()
        project_step = 1 if self.mode == 'squalane' else 0
        if system is None and self.steps[self.index].check != 'download':
            self.index = project_step
        step = self.steps[self.index]
        stages = system.stages if system is not None else {}
        ready = system is not None
        complete = False
        note = ''
        if step.check == 'download':
            ready = complete = True
            note = 'Already have the files? Click Next.'
        elif self.index == project_step:
            complete = ready
        elif not ready:
            pass
        elif step.prerequisite and stages.get(step.prerequisite) != 'complete':
            ready = False
            note = 'Finish {} first.'.format(step.prerequisite.capitalize())
        elif step.stage and stages.get(step.stage) == 'complete':
            complete = True
            note = ''
        elif step.check:
            complete = tutorial_check(step.check, system)
            ready = complete or step.optional
            note = ''
        elif step.optional:
            note = ''
        elif step.stage:
            ready = False
            status = stages.get(step.stage, 'pending')
            note = 'Failed. Review the log and retry.' if status == 'failed' else ''
        else:
            complete = True
            note = ''
        self.progress.setText('Step {} of {}'.format(self.index + 1, len(self.steps)))
        self.title.setText(step.title)
        # A completed action must not continue telling the user to run it again.
        self.text.setText(step.done_text if complete and (step.stage or step.check not in (None, 'download'))
                          else step.text)
        self.context_label.setText('System: {}'.format(system.name) if system else '')
        self.context_label.setVisible(system is not None)
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.download_row.setVisible(step.check == 'download')
        self.back_button.setEnabled(self.index > 0 and not instruction_modal)
        self.next_button.setEnabled(ready and not instruction_modal)
        self.next_button.setText('Finish' if self.index == len(self.steps) - 1 else
                                 'Skip' if step.optional and not complete else 'Next')
        target = self.window
        for name in step.target.split('.') if step.target else ():
            target = getattr(target, name, None)
            if target is None:
                break
        if not step.target:
            target = None
        if instruction_modal:
            target = getattr(modal, 'canvas', modal)
        on_tab = self.window.tabs.currentIndex() == step.tab
        self._highlight(target if target is not None and on_tab and target.isVisible() else None)
        if not self.dialog.isVisible():
            self.dialog.show()

    def _highlight(self, widget):
        if widget is self.highlighted:
            return
        previous = self.highlighted
        self.highlighted = widget
        for target, enabled in ((previous, False), (widget, True)):
            if target is not None:
                try:
                    target.setProperty('tutorialTarget', enabled)
                    target.style().unpolish(target)
                    target.style().polish(target)
                    target.update()
                except RuntimeError:
                    # A previously highlighted popup may already be destroyed.
                    pass
