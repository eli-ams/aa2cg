from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from rdkit import Chem

from . import APP_TITLE
from .backend.validation_metrics import load_property_statistics
from .bead_colors import bead_color
from .fit_settings import normalize_override, resolve_override
from .molecule_drawer import MoleculeCanvas
from .optimization_history import best_directory
from .project_export_dialog import ProjectExportDialog
from .validation_sampling import normalize_sampling
from .pages import (
    BeadRuleDialog,
    ForceMatchPage,
    ImportPage,
    LammpsPage,
    MappingPage,
    MergePage,
    OptimizePage,
    TablesPage,
    ValidationPage,
)
from .process_runner import ProcessRunner
from .workflow_views import DistributionPage
from .project import PROJECT_FILENAME, Project, display_command, split_command
from .tutorial import StartupDialog, TutorialController, onboarding_settings


STAGE_ORDER = ["inspect", "cgmodel", "force", "postprocess", "lammps", "optimize", "validate"]

STAGE_LABELS = {
    "inspect": "Inspect",
    "cgmodel": "Build CG model",
    "force": "Force match",
    "postprocess": "Prepare tables",
    "lammps": "Build LAMMPS system",
    "smoke": "LAMMPS smoke test",
    "optimize": "Refine (optional)",
    "merge": "Merge force fields",
    "validate": "Validate",
}


class ProcessDialog(QtWidgets.QDialog):
    """Application-modal monitor for one background workflow stage."""

    cancelRequested = QtCore.Signal()

    def __init__(self, stage: str, system_name: str, project_path: Path, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.system_name = system_name
        self._running = True
        self._cancel_requested = False
        self._status_text = "Running"
        self.setWindowTitle(f"Running · {STAGE_LABELS.get(stage, stage.title())}")
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.resize(900, 650)
        self.setMinimumSize(680, 460)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

        root = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(STAGE_LABELS.get(stage, stage.title()))
        heading.setObjectName("pageTitle")
        context = QtWidgets.QLabel(f"{system_name}  ·  {project_path}")
        context.setObjectName("pageDescription")
        context.setWordWrap(True)
        root.addWidget(heading)
        root.addWidget(context)

        progress_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Starting…")
        self.status_label.setObjectName("sectionTitle")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        progress_row.addWidget(self.status_label)
        progress_row.addWidget(self.progress, 1)
        root.addLayout(progress_row)

        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(8000)
        self.console.setObjectName("console")
        root.addWidget(self.console, 1)

        actions = QtWidgets.QHBoxLayout()
        self.copy_button = QtWidgets.QPushButton("Copy log")
        self.action_button = QtWidgets.QPushButton("Cancel run")
        actions.addWidget(self.copy_button)
        actions.addStretch()
        actions.addWidget(self.action_button)
        root.addLayout(actions)

        self.copy_button.clicked.connect(self._copy_log)
        self.action_button.clicked.connect(self._cancel_or_close)
        self.elapsed = QtCore.QElapsedTimer()
        self.elapsed.start()
        self.elapsed_timer = QtCore.QTimer(self)
        self.elapsed_timer.setInterval(1000)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.elapsed_timer.start()
        self._update_elapsed()

    @property
    def running(self) -> bool:
        return self._running

    def append_log(self, text: str):
        cursor = self.console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def finish_run(self, success: bool):
        self._running = False
        self.elapsed_timer.stop()
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if success else 0)
        if success:
            self._status_text = "Completed"
            self._update_elapsed()
            self.action_button.setEnabled(False)
            QtCore.QTimer.singleShot(350, self.accept)
            return

        self._status_text = "Cancelled" if self._cancel_requested else "Failed"
        self._update_elapsed()
        self.action_button.setText("Close")
        self.action_button.setEnabled(True)
        self.hide()
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setModal(False)
        self.show()
        self.raise_()
        self.activateWindow()

    def _update_elapsed(self):
        seconds = max(0, self.elapsed.elapsed() // 1000)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_text = (
            f"{hours:d}:{minutes:02d}:{seconds:02d}"
            if hours
            else f"{minutes:02d}:{seconds:02d}"
        )
        self.status_label.setText(f"{self._status_text}  ·  {elapsed_text}")

    def _copy_log(self):
        QtWidgets.QApplication.clipboard().setText(self.console.toPlainText())

    def _cancel_or_close(self):
        if not self._running:
            self.close()
            return
        if self._cancel_requested:
            return
        self._cancel_requested = True
        self._status_text = "Cancelling"
        self._update_elapsed()
        self.action_button.setEnabled(False)
        self.cancelRequested.emit()

    def closeEvent(self, event):
        if self._running:
            event.ignore()
            return
        super().closeEvent(event)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self.current_system_id: str | None = None
        self._loading_canvas = False
        self._bead_dialog: BeadRuleDialog | None = None
        self._process_dialog: ProcessDialog | None = None
        self._export_dialog = None
        self._log_buffer = ""
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 920)
        self.setMinimumSize(1100, 700)

        self.runner = ProcessRunner(self)
        self.runner.outputReceived.connect(self._append_log)
        self.runner.finished.connect(self._stage_finished)

        self._build_actions()
        self._build_ui()
        self._connect_species_preview()
        self._set_project_enabled(False)
        self.tutorial = TutorialController(self)
        self._welcome_dialog = None

    def _build_actions(self):
        self.new_action = QtGui.QAction("New project…", self)
        self.open_action = QtGui.QAction("Open project…", self)
        self.save_action = QtGui.QAction("Save", self)
        self.export_action = QtGui.QAction('Export project…', self)
        self.open_folder_action = QtGui.QAction("Open project folder", self)
        self.quit_action = QtGui.QAction("Quit", self)
        self.new_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.new_action.triggered.connect(self.new_project)
        self.open_action.triggered.connect(self.open_project)
        self.save_action.triggered.connect(self.save_project)
        self.export_action.triggered.connect(self.export_project)
        self.open_folder_action.triggered.connect(self.open_project_folder)
        self.quit_action.triggered.connect(self.close)
        menu = self.menuBar().addMenu("File")
        menu.addActions([self.new_action, self.open_action, self.save_action, self.export_action])
        menu.addSeparator()
        menu.addAction(self.open_folder_action)
        menu.addSeparator()
        menu.addAction(self.quit_action)
        help_menu = self.menuBar().addMenu('Help')
        self.welcome_action = help_menu.addAction('Welcome…')
        self.welcome_action.triggered.connect(self.show_welcome)
        self.guide_action = help_menu.addAction('Guided introduction…')
        self.guide_action.triggered.connect(self.start_guided_introduction)
        self.example_action = help_menu.addAction('Squalane tutorial…')
        self.example_action.triggered.connect(self.start_squalane_tutorial)

    def show_welcome_on_startup(self):
        if (self.project is None and
                onboarding_settings().value('show_welcome', True, type=bool)):
            self.show_welcome()

    def show_welcome(self):
        if self.runner.running or self._welcome_dialog is not None:
            return
        dialog = StartupDialog(self)
        self._welcome_dialog = dialog
        try:
            accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
            onboarding_settings().setValue('show_welcome', dialog.show_on_startup.isChecked())
            choice = dialog.choice if accepted else ''
        finally:
            self._welcome_dialog = None
            dialog.deleteLater()
        if choice == 'new':
            self.new_project()
        elif choice == 'open':
            self.open_project()
        elif choice == 'tutorial':
            self.start_squalane_tutorial()

    def start_guided_introduction(self):
        if not self.runner.running:
            self.tutorial.start()

    def start_squalane_tutorial(self):
        if not self.runner.running:
            self.tutorial.start(mode='squalane')

    def _build_ui(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_system_bar())
        root.addWidget(self._build_workflow_panel(), 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Create or open a project to begin")

    def _build_system_bar(self):
        panel = QtWidgets.QFrame()
        panel.setObjectName("systemBar")
        layout = QtWidgets.QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        label = QtWidgets.QLabel("Systems")
        label.setObjectName("sectionTitle")
        self.add_system_button = QtWidgets.QPushButton("＋ Add system")
        self.add_system_button.setToolTip("Add system")
        self.remove_system_button = QtWidgets.QPushButton("− Remove")
        self.remove_system_button.setToolTip("Remove selected system")
        self.add_system_button.clicked.connect(self.add_system)
        self.remove_system_button.clicked.connect(self.remove_system)
        self.system_list = QtWidgets.QListWidget()
        self.system_list.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.system_list.setWrapping(False)
        self.system_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.system_list.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.system_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.system_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.system_list.setSpacing(3)
        self.system_list.setFixedHeight(44)
        self.system_list.currentItemChanged.connect(self._system_changed)
        layout.addWidget(label)
        layout.addWidget(self.system_list, 1)
        layout.addWidget(self.add_system_button)
        layout.addWidget(self.remove_system_button)
        self.combine_button = QtWidgets.QPushButton('Combine systems…')
        self.combine_button.clicked.connect(self._open_combine)
        layout.addWidget(self.combine_button)
        return panel

    def _build_species_preview_panel(self):
        panel = QtWidgets.QFrame()
        panel.setObjectName("speciesPreview")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        self.canvas_mode_label = QtWidgets.QLabel("Molecular species")
        self.canvas_mode_label.setObjectName("sectionTitle")
        self.species_selector = QtWidgets.QComboBox(panel)
        self.species_selector.setPlaceholderText("Run Stage 1 to detect species")
        self.species_selector.currentIndexChanged.connect(self._species_selection_changed)
        self.mapping_status_panel = QtWidgets.QFrame(panel)
        self.mapping_status_panel.setObjectName("mappingStatusPanel")
        status_layout = QtWidgets.QHBoxLayout(self.mapping_status_panel)
        status_layout.setContentsMargins(6, 3, 6, 3)
        status_layout.setSpacing(8)
        self.mapping_state_badge = QtWidgets.QLabel("Not mapped")
        self.mapping_state_badge.setObjectName("stageStatus")
        self.mapping_selected_label = QtWidgets.QLabel("Selected: 0")
        self.mapping_saved_label = QtWidgets.QLabel("Types: 0")
        self.mapping_saved_label.setToolTip("Number of saved bead types for this species")
        self.mapping_mapped_label = QtWidgets.QLabel("Mapped: 0/0")
        self.mapping_unmapped_label = QtWidgets.QLabel("Unmapped: 0")
        self.mapping_matches_label = QtWidgets.QLabel("Matches: 0")
        self.clear_mapping_selection_button = QtWidgets.QPushButton("Clear selection")
        self.clear_mapping_selection_button.clicked.connect(self._clear_mapping_selection)
        status_layout.addWidget(self.mapping_state_badge)
        status_layout.addWidget(self.mapping_saved_label)
        status_layout.addWidget(self.mapping_mapped_label)
        status_layout.addWidget(self.mapping_unmapped_label)
        status_layout.addStretch()
        # Retain the legacy selection controls, hidden in the preview workspace.
        for widget in (self.mapping_selected_label, self.mapping_matches_label, self.clear_mapping_selection_button):
            widget.setParent(self.mapping_status_panel)
        self.mapping_selected_label.hide()
        self.mapping_matches_label.hide()
        self.clear_mapping_selection_button.hide()
        self.mapping_status_panel.hide()
        self.canvas = MoleculeCanvas()
        self.canvas.setMinimumSize(220, 70)
        self.canvas.installEventFilter(self)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.canvas_summary = QtWidgets.QLabel("Run Stage 1 to inspect molecular species", panel)
        self.canvas_summary.setObjectName("canvasSummary")
        self.canvas_summary.setWordWrap(True)
        layout.addWidget(self.canvas, 1)
        canvas_toolbar = QtWidgets.QHBoxLayout()
        canvas_toolbar.setSpacing(6)
        self.canvas_species_button = QtWidgets.QToolButton(text='Species ▾')
        self.canvas_species_button.setToolTip('Choose the molecular species shown in this canvas')
        self.canvas_species_button.clicked.connect(self._show_canvas_species_menu)
        canvas_toolbar.addWidget(self.mapping_page.open_bead_editor_button)
        self.mapping_page.open_bead_editor_button.hide()
        canvas_toolbar.addWidget(self.canvas_species_button)
        canvas_toolbar.addStretch(1)
        canvas_toolbar.addWidget(self.mapping_state_badge)
        self.mapping_state_badge.hide()
        self.canvas_view_button = QtWidgets.QToolButton(text='View ▾')
        self.canvas_view_button.clicked.connect(self._show_canvas_view_menu)
        canvas_toolbar.addWidget(self.canvas_view_button)
        layout.addLayout(canvas_toolbar)
        # Retain the selector and detail models without allocating canvas space.
        self.canvas_mode_label.setParent(panel)
        for widget in (self.species_selector, self.canvas_mode_label, self.canvas_summary):
            widget.hide()
        return panel

    def _show_canvas_species_menu(self):
        menu = QtWidgets.QMenu(self)
        group = QtGui.QActionGroup(menu)
        for index in range(self.species_selector.count()):
            action = menu.addAction(self.species_selector.itemText(index))
            action.setCheckable(True)
            action.setChecked(index == self.species_selector.currentIndex())
            group.addAction(action)
            action.triggered.connect(lambda checked=False, i=index: self.species_selector.setCurrentIndex(i))
        if not self.species_selector.count():
            menu.addAction('Inspect inputs to detect species').setEnabled(False)
        menu.exec(self.canvas_species_button.mapToGlobal(QtCore.QPoint(0, self.canvas_species_button.height())))
        menu.deleteLater()

    def _show_canvas_view_menu(self):
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.canvas_mode_label.text()).setEnabled(False)
        menu.addAction('Fit molecule to canvas', self._fit_species_canvas)
        menu.addAction('Species and mapping details…', self._show_canvas_details)
        if self.tabs.currentIndex() == 1:
            menu.addSeparator()
            menu.addAction('Mapping settings and saved rules…', self._show_mapping_settings)
        menu.exec(self.canvas_view_button.mapToGlobal(QtCore.QPoint(0, self.canvas_view_button.height())))
        menu.deleteLater()

    def _show_mapping_settings(self):
        self.mapping_page.settings_dialog.show()
        self.mapping_page.settings_dialog.raise_()
        self.mapping_page.settings_dialog.activateWindow()

    def _show_canvas_details(self):
        text = self.canvas_summary.text()
        if self.mapping_state_badge.isVisible():
            text += '\n\n' + '\n'.join(label.text() for label in (
                self.mapping_state_badge, self.mapping_saved_label,
                self.mapping_mapped_label, self.mapping_unmapped_label,
            ))
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle('Species and mapping details')
        dialog.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        dialog.setText(text)
        dialog.exec()

    def _build_workflow_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 4, 8, 4)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.import_page = ImportPage()
        self.mapping_page = MappingPage()
        self.force_page = ForceMatchPage()
        self.post_page = TablesPage()
        self.lammps_page = LammpsPage()
        self.optimize_page = OptimizePage()
        self.optimize_page.settingsChanged.connect(self._optimization_settings_changed)
        self.optimize_page.continueRequested.connect(lambda explore, refine: self.run_stage('optimize', (explore, refine)))
        self.merge_page = MergePage()
        self.validation_page = ValidationPage()
        self.distribution_page = DistributionPage()
        self.distribution_page.overrideChanged.connect(self._save_fit_override)
        self.post_page.explorer.overrideChanged.connect(self._save_table_override)
        self.distribution_page.continueRequested.connect(lambda: self.tabs.setCurrentIndex(3))
        self.combine_dialog = QtWidgets.QDialog(self)
        self.combine_dialog.setWindowTitle('Combine systems')
        self.combine_dialog.resize(1000, 760)
        QtWidgets.QVBoxLayout(self.combine_dialog).addWidget(self.merge_page)
        self.merge_page.heading.setText('Combine systems')
        self.merge_page.runRequested.connect(lambda _: self.combine_dialog.hide())
        for title, page in [
            ("1  Inspect", self.import_page),
            ("2  Map", self.mapping_page),
            ("3  Configure fit", self.distribution_page),
            ("4  Fit", self.force_page),
            ("5  Prepare tables", self.post_page),
            ("6  LAMMPS", self.lammps_page),
            ("7  Refine (optional)", self.optimize_page),
            ("8  Validate", self.validation_page),
        ]:
            self.tabs.addTab(page, title)
        # Navigation lives in the existing header; it never starts a job.
        for page, target, label, guidance in [
            (self.import_page, self.mapping_page, 'Map', 'After inspecting inputs and resolving failed checks, define the bead mapping.'),
            (self.mapping_page, self.distribution_page, 'Configure fit', 'After building the mapped trajectory, review the histogram-based fit settings.'),
            (self.force_page, self.post_page, 'Prepare tables', 'After fitting, compare raw and processed curves and export the prepared tables.'),
            (self.post_page, self.lammps_page, 'LAMMPS', 'After preparing tables, generate and smoke-test the LAMMPS setup.'),
            (self.optimize_page, self.validation_page, 'Validate', 'Compare an existing CG trajectory with the AA reference. Refinement is optional.'),
        ]:
            button = QtWidgets.QToolButton(text='Next: ' + label)
            button.setToolTip(guidance + ' Opens the tab only; no calculation is started.')
            button.clicked.connect(lambda checked=False, target=target: self.tabs.setCurrentWidget(target))
            page.header_layout.addWidget(button)
        next_button = QtWidgets.QToolButton(text='Next…')
        next_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        next_button.setToolTip('Refine is optional. Validate requires a CG simulation trajectory, not just a successful smoke test.')
        next_menu = QtWidgets.QMenu(next_button)
        next_menu.addAction('7 · Refine (optional)', lambda: self.tabs.setCurrentWidget(self.optimize_page))
        next_menu.addAction('8 · Validate existing CG simulation', lambda: self.tabs.setCurrentWidget(self.validation_page))
        next_button.setMenu(next_menu)
        self.lammps_page.header_layout.addWidget(next_button)
        for page in [
            self.import_page,
            self.mapping_page,
            self.force_page,
            self.post_page,
            self.lammps_page,
            self.optimize_page,
            self.merge_page,
            self.validation_page,
        ]:
            page.runRequested.connect(self.run_stage)
        self.import_page.browseTrajectoryRequested.connect(self.browse_trajectory)
        self.import_page.browseStructureRequested.connect(self.browse_structure)
        self.import_page.speciesPreviewRequested.connect(self.preview_species)
        self.mapping_page.speciesPreviewRequested.connect(self.preview_species)
        self.mapping_page.customRulePreviewRequested.connect(self._preview_custom_mapping_rule)
        self.mapping_page.customRuleAccepted.connect(self._save_custom_mapping_rule)
        self.mapping_page.customRuleCancelled.connect(self._cancel_custom_mapping_rule)
        self.mapping_page.beadCreationModeChanged.connect(self._bead_creation_mode_changed)
        self.mapping_page.customRulesClearRequested.connect(self._clear_current_species_rules)
        self.mapping_page.customRuleDeleteRequested.connect(
            self._delete_selected_custom_rule
        )
        self.mapping_page.customRuleCreationRequested.connect(self._open_custom_mapping_dialog)
        self.mapping_page.descriptorSettingsChanged.connect(self._preview_custom_mapping_rule)
        self.mapping_page.mappingModeChanged.connect(self._render_selected_species)
        self.validation_page.aaPropertiesRequested.connect(self.browse_aa_properties)
        self.validation_page.samplingChanged.connect(self._validation_sampling_changed)
        self.lammps_page.smokeTestRequested.connect(self.run_smoke_test)
        self.lammps_page.commandChanged.connect(self._lammps_command_changed)
        self.import_page.name.editingFinished.connect(self._commit_current_system)
        self.import_page.inputs_dialog.finished.connect(self.save_project)
        self.tabs.currentChanged.connect(self._workflow_tab_changed)

        self.species_panel = self._build_species_preview_panel()
        self.import_page.preview_layout.addWidget(self.species_panel, 1)
        self.trajectory_views = [self.import_page.trajectory_view, self.mapping_page.trajectory_view,
                                 self.validation_page.trajectory_view, self.validation_page.cg_view]
        # The comparison owns CG selection; update its AA side only to avoid
        # feedback between global species selection and the paired controls.
        for view in self.trajectory_views[:-1]:
            view.focusChanged.connect(self._focus_molecule)
        # Inspect and Map share AA frame indices. Validation has its own paired
        # relative-progress controller and must not receive raw frame indices.
        for view in self.trajectory_views[:2]:
            view.frameChanged.connect(self._sync_aa_frame)
        self.mapping_page.settings_dialog.finished.connect(self._commit_current_system)
        layout.addWidget(self.tabs, 1)
        return panel

    def _sync_aa_frame(self, index):
        for view in self.trajectory_views[:2]:
            if view is not self.sender():
                view.set_frame(index)

    def _open_combine(self):
        if self.project is None or self.runner.running:
            return
        self._commit_current_system()
        self.merge_page.load_project(self.project)
        self.combine_dialog.show()
        self.combine_dialog.raise_()

    def _focus_molecule(self, identity, instance):
        if getattr(self, '_syncing_focus', False):
            return
        self._syncing_focus = True
        try:
            for view in self.trajectory_views[:-1]:
                if view is not self.sender():
                    view.set_focus(identity, instance)
            if self.project and self.current_system_id:
                system = self.project.system(self.current_system_id)
                for index, species in enumerate(system.detected_species):
                    if species.get('id') == identity:
                        self.species_selector.setCurrentIndex(index)
                        break
        finally:
            self._syncing_focus = False

    def _sync_visuals(self):
        if not self.project or not self.current_system_id:
            return
        system = self.project.system(self.current_system_id)
        root = self.project.system_root(system)
        for view in self.trajectory_views[:-1]:
            view.set_source(system, self.project.root)
        trajectory = self.project.validation_trajectory(system)
        structure_candidates = ([trajectory.parent / 'structure.data'] if trajectory else []) + [root / 'output' / 'lammps' / 'structure.data']
        structure = next((p for p in structure_candidates if p.is_file()), None)
        self.validation_page.cg_view.set_source(system, self.project.root,
                                                cg_structure=structure, cg_trajectory=trajectory)
        self.distribution_page.load_system_root(
            root, system.stages.get('cgmodel') == 'complete', system.fit_overrides)

    def _save_fit_override(self, name, override):
        if not self.project or not self.current_system_id or self.runner.running:
            return
        system = self.project.system(self.current_system_id)
        root = self.project.system_root(system)
        if root != self.distribution_page.root or system.stages.get('cgmodel') != 'complete':
            return
        old_overrides = deepcopy(system.fit_overrides)
        old_stages = dict(system.stages)
        old_invalidated = list(system.invalidated_stages)
        old_merge_status = self.project.merge_status
        old_merge_invalidated = self.project.merge_invalidated
        try:
            entry = next(item for item in self.distribution_page.entries if item['name'] == name)
            if override is None:
                system.fit_overrides.pop(name, None)
            else:
                resolve_override(entry, override, root=root)
                system.fit_overrides[name] = normalize_override(entry, override)
            if system.fit_overrides == old_overrides:
                return
            self._reset_from(system, 'force')
            self.project.save()
        except Exception as exc:
            system.fit_overrides = old_overrides
            system.stages = old_stages
            system.invalidated_stages = old_invalidated
            self.project.merge_status = old_merge_status
            self.project.merge_invalidated = old_merge_invalidated
            self.distribution_page._key = None
            self.distribution_page.load_system_root(root, True, old_overrides)
            self._message('Could not save fitting settings', str(exc), critical=True)
            return
        # Do not reload the editor mid-drag or discard the user's current zoom.
        self.force_page.review.load(root, system.fit_overrides,
                                    system.stages.get('cgmodel') == 'complete',
                                    system.stages.get('force', 'pending'),
                                    self.force_page.method.currentData())
        for page, stage in ((self.force_page, 'force'), (self.post_page, 'postprocess'),
                            (self.lammps_page, 'lammps'), (self.optimize_page, 'optimize'),
                            (self.validation_page, 'validate')):
            page.set_status(system.stages[stage])
        self.lammps_page.set_smoke_status(system.stages['smoke'])
        self.merge_page.set_status(self.project.merge_status)
        self.statusBar().showMessage('Fitting settings saved. Run Fit to update this system and downstream results.', 8000)

    def _save_table_override(self, name, override):
        if not self.project or not self.current_system_id or self.runner.running:
            return
        system = self.project.system(self.current_system_id)
        root = self.project.system_root(system)
        editor = self.post_page.explorer
        if root != editor.system_root:
            return
        old_overrides = deepcopy(system.table_overrides)
        old_stages, old_invalidated = dict(system.stages), list(system.invalidated_stages)
        old_merge, old_merge_invalidated = self.project.merge_status, self.project.merge_invalidated
        try:
            if override is None:
                system.table_overrides.pop(name, None)
            else:
                from .backend.table_processing import normalize_processing_settings
                from .results_viewers import read_table_rows
                if Path(name).name != name or not name.endswith('.table'):
                    raise ValueError('Invalid table name.')
                raw = read_table_rows(root / 'output' / 'tables' / name)
                system.table_overrides[name] = normalize_processing_settings(name.split('_')[0], override, raw[0])
            if old_overrides == system.table_overrides:
                return
            self._reset_from(system, 'postprocess')
            self.project.save()
        except Exception as exc:
            system.table_overrides = old_overrides
            system.stages, system.invalidated_stages = old_stages, old_invalidated
            self.project.merge_status, self.project.merge_invalidated = old_merge, old_merge_invalidated
            editor.load_system_root(root, old_overrides, old_stages.get('postprocess') == 'complete')
            self._message('Could not save table settings', str(exc), critical=True)
            return
        for page, stage in ((self.post_page, 'postprocess'), (self.lammps_page, 'lammps'),
                            (self.optimize_page, 'optimize'), (self.validation_page, 'validate')):
            page.set_status(system.stages[stage])
        self.lammps_page.set_smoke_status(system.stages['smoke'])
        self.merge_page.set_status(self.project.merge_status)
        self.statusBar().showMessage('Table settings saved. Run Prepare tables to export and update downstream results.', 8000)

    def _connect_species_preview(self):
        self.canvas.changed.connect(self._canvas_changed)
        self.canvas.set_selection_only(False)
        self.canvas.set_read_only(True)
        self.canvas.set_united_atom_mode(False)

    def eventFilter(self, watched, event):
        if watched is getattr(self, 'canvas', None) and event.type() == QtCore.QEvent.Type.Resize:
            QtCore.QTimer.singleShot(0, self._fit_species_canvas)
        return super().eventFilter(watched, event)

    def _fit_species_canvas(self):
        if not self.canvas.model.atoms:
            return
        xs, ys = zip(*[(a[0], a[1]) for a in self.canvas.model.atoms.values()])
        scale = min(1.0, max(1, self.canvas.width() - 60) / max(1, max(xs) - min(xs)),
                    max(1, self.canvas.height() - 60) / max(1, max(ys) - min(ys)))
        self.canvas.scale = scale
        self.canvas.pan_x = self.canvas.width() / 2 - (max(xs) + min(xs)) / 2 * scale
        self.canvas.pan_y = self.canvas.height() / 2 - (max(ys) + min(ys)) / 2 * scale
        self.canvas.update()

    def _place_species_preview(self):
        index = self.tabs.currentIndex()
        self.mapping_page.open_bead_editor_button.setVisible(index == 1)
        if index == 0:
            target = self.import_page.preview_layout
        elif index == 1:
            target = self.mapping_page.preview_layout
        else:
            self.species_panel.hide()
            return
        if self.species_panel.parentWidget() is not target.parentWidget():
            target.addWidget(self.species_panel, 1)
        self.species_panel.show()

    def new_project(self):
        if self.runner.running:
            self._message("A stage is running", "Cancel the running stage before changing projects.")
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a new project folder")
        if not directory:
            return
        path = Path(directory) / PROJECT_FILENAME
        if path.exists():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Project already exists",
                "This folder already contains an OpenMSCG project. Open it?",
            )
            if answer == QtWidgets.QMessageBox.StandardButton.Yes:
                self._load_project(path)
            return
        project = Project.create(directory)
        project.add_system("molecule")
        self._set_project(project)

    def open_project(self):
        if self.runner.running:
            self._message("A stage is running", "Cancel the running stage before changing projects.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open OpenMSCG project",
            "",
            f"OpenMSCG project ({PROJECT_FILENAME});;JSON files (*.json)",
        )
        if path:
            self._load_project(Path(path))

    def _load_project(self, path: Path):
        try:
            project = Project.load(path)
            self._hydrate_statuses(project)
            self._set_project(project)
        except Exception as exc:
            self._message("Could not open project", str(exc), critical=True)

    def _set_project(self, project: Project):
        for view in self.trajectory_views:
            view.clear()
        self.distribution_page.clear()
        self.project = project
        self.current_system_id = None
        self.setWindowTitle(f'{project.name} — {APP_TITLE}')
        self.open_folder_action.setToolTip(str(project.root))
        self._set_project_enabled(True)
        self.optimize_page.load_settings(project.execution)
        self.merge_page.load_project(project)
        self._refresh_system_list()
        if self.system_list.count():
            self.system_list.setCurrentRow(0)
        self._append_log(f"\nProject opened: {project.path}\n")
        self.statusBar().showMessage("Project ready")

    def _set_project_enabled(self, enabled: bool):
        self.save_action.setEnabled(enabled)
        self.export_action.setEnabled(enabled)
        self.open_folder_action.setEnabled(enabled)
        self.add_system_button.setEnabled(enabled)
        self.remove_system_button.setEnabled(enabled)
        self.combine_button.setEnabled(enabled)
        self.system_list.setEnabled(enabled)
        self.tabs.setEnabled(enabled)

    def _refresh_system_list(self):
        if self.project is None:
            return
        selected = self.current_system_id
        self.system_list.blockSignals(True)
        self.system_list.clear()
        for system in self.project.systems:
            completed = sum(system.stages.get(stage) == "complete" for stage in STAGE_ORDER)
            item = QtWidgets.QListWidgetItem(f"{system.name}    {completed}/{len(STAGE_ORDER)}")
            item.setSizeHint(
                QtCore.QSize(
                    max(170, self.system_list.fontMetrics().horizontalAdvance(item.text()) + 28),
                    32,
                )
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, system.id)
            self.system_list.addItem(item)
            if system.id == selected:
                self.system_list.setCurrentItem(item)
        self.system_list.blockSignals(False)

    def add_system(self):
        if self.project is None:
            return
        self._commit_current_system()
        system = self.project.add_system(f"molecule {len(self.project.systems) + 1}")
        self.current_system_id = system.id
        self._refresh_system_list()
        for index in range(self.system_list.count()):
            item = self.system_list.item(index)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == system.id:
                self.system_list.setCurrentItem(item)
                self._load_system(system)
                break
        self.merge_page.load_project(self.project)

    def remove_system(self):
        if self.project is None or self.current_system_id is None:
            return
        system = self.project.system(self.current_system_id)
        answer = QtWidgets.QMessageBox.question(
            self,
            "Remove system",
            f"Remove {system.name!r} from this project? Existing output files are preserved.",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.project.remove_system(system.id)
        self.current_system_id = None
        self._refresh_system_list()
        if self.system_list.count():
            self.system_list.setCurrentRow(0)
        else:
            self.canvas.clear_all()
            for view in self.trajectory_views:
                view.clear()
            self.distribution_page.clear()
            self.import_page.load_system(None)
            self.mapping_page.load_system(None)
            self.force_page.load_system(None)
            self.post_page.load_system_root(None)
            self.lammps_page.load_system_root(None)
            self.validation_page.load_system(None)
        self.merge_page.load_project(self.project)

    def _system_changed(self, current, _previous):
        if current is None or self.project is None:
            return
        self._commit_current_system()
        system_id = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self.current_system_id = system_id
        self._load_system(self.project.system(system_id))

    def _load_system(self, system):
        self.import_page.load_system(system)
        self.mapping_page.load_system(system)
        self.mapping_page.set_status(system.stages.get("cgmodel", "pending"))
        root = self.project.system_root(system)
        self.force_page.load_system(system, root)
        self.force_page.set_status(system.stages.get("force", "pending"))
        self.post_page.set_status(system.stages.get("postprocess", "pending"))
        self.lammps_page.load_system_root(root, self.project.execution.lammps_command)
        self.lammps_page.set_status(system.stages.get("lammps", "pending"))
        self.lammps_page.set_smoke_status(system.stages.get("smoke", "pending"))
        self.optimize_page.load_settings(self.project.execution, system, root)
        self.optimize_page.set_status(system.stages.get("optimize", "pending"))
        self.post_page.load_system_root(root, system.table_overrides, system.stages.get('postprocess') == 'complete')
        self.validation_page.load_system(system, root)
        self.validation_page.set_status(system.stages.get("validate", "pending"))
        self._refresh_species_selector(system)

        self._sync_visuals()

    def _commit_current_system(self):
        if self.project is None or self.current_system_id is None:
            return
        try:
            system = self.project.system(self.current_system_id)
        except KeyError:
            return
        previous_structure = system.structure_file
        previous_structure_inferred = system.structure_inferred
        previous_trajectory = system.trajectory
        previous_name = system.name
        previous_species = deepcopy(system.detected_species)
        previous_mapping = (
            system.interaction_level,
            system.mapping_mode,
            deepcopy(system.custom_mapping_rules),
        )
        previous_force_match_method = system.force_match_method
        self.import_page.apply(system)
        self.mapping_page.apply(system)
        self.force_page.apply(system)
        if previous_trajectory != system.trajectory and previous_structure_inferred:
            system.structure_file = ""
            system.structure_inferred = False
            system.infer_bonds_from_trajectory = False
            system.detected_species = []
            system.trajectory_stats = {}
            system.smiles = ""
            system.molecule_count = 1
            self.import_page.structure_file.clear()
            self.import_page._load_species([])
            self.import_page._load_stats({})
        elif previous_structure != system.structure_file:
            system.structure_inferred = False
            system.infer_bonds_from_trajectory = False
        if previous_structure != system.structure_file and system.detected_species:
            system.detected_species = []
            system.trajectory_stats = {}
            system.smiles = ""
            system.molecule_count = 1
            self.import_page._load_species([])
            self.import_page._load_stats({})
        input_changed = (
            previous_structure != system.structure_file
            or previous_trajectory != system.trajectory
        )
        if input_changed:
            system.trajectory_stats = {}
            self.import_page._load_stats({})
            self._reset_from(system, "inspect")
        elif previous_name != system.name or previous_species != system.detected_species:
            self._reset_from(system, "cgmodel")
        if previous_mapping != (
            system.interaction_level,
            system.mapping_mode,
            system.custom_mapping_rules,
        ):
            self._reset_from(system, "cgmodel")
        elif previous_force_match_method != system.force_match_method:
            self._reset_from(system, "force")
        old_objective = deepcopy(system.optimization_properties)
        old_protocol = deepcopy(system.optimization_protocol)
        old_search = deepcopy(system.optimization_search)
        old_execution = (
            self.project.execution.optimize_trials,
            list(self.project.execution.lammps_command),
            self.project.execution.include_structural_loss,
        )
        self.optimize_page.apply_settings(self.project.execution, system)
        new_execution = (
            self.project.execution.optimize_trials,
            list(self.project.execution.lammps_command),
            self.project.execution.include_structural_loss,
        )
        if (old_objective != system.optimization_properties or old_execution != new_execution
                or old_protocol != system.optimization_protocol
                or old_search != system.optimization_search):
            self._reset_from(system, "optimize")
        self.merge_page.apply(self.project)
        self.project.save()
        self.setWindowTitle(f'{self.project.name} — {APP_TITLE}')
        self.open_folder_action.setToolTip(str(self.project.root))
        self.import_page.update_input_summary(system)
        self.import_page.set_status(system.stages.get('inspect', 'pending'), self.runner.running)
        self._sync_visuals()

    def _optimization_settings_changed(self):
        self.save_project()
        self._refresh_pages()

    def export_project(self):
        if self.project is None:
            return
        if self.runner.running:
            self._message('Workflow busy', 'Wait for or cancel the current job before exporting.')
            return
        try:
            self._commit_current_system()
            dialog = ProjectExportDialog(self.project, self)
            self._export_dialog = dialog
            dialog.exported.connect(lambda path: self.statusBar().showMessage('Exported: ' + path, 10000))
            dialog.exec()
            dialog.deleteLater()
        except Exception as exc:
            self._message('Could not export project', str(exc), critical=True)
        finally:
            self._export_dialog = None

    def save_project(self):
        try:
            self._commit_current_system()
            self._refresh_system_list()
            self.statusBar().showMessage("Project saved", 3000)
        except Exception as exc:
            self._message("Could not save project", str(exc), critical=True)

    def browse_trajectory(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.import_page.inputs_dialog,
            "Select atomistic trajectory",
            "",
            "LAMMPS trajectories (*.lammpstrj *.dump *.traj);;All files (*)",
        )
        if path:
            self.import_page.trajectory.setText(path)
            if not self.import_page.structure_file.text().strip():
                for filename in ("structure.data", "structure.dat", "aa_structure.data", "aa_structure.dat"):
                    sibling = Path(path).parent / filename
                    if sibling.is_file():
                        self.import_page.structure_file.setText(str(sibling))
                        break
            self._commit_current_system()
            selected_structure = self.import_page.structure_file.text().strip()
            if not selected_structure or not Path(selected_structure).is_file():
                system = self.project.system(self.current_system_id)
                self._prepare_inspection_source(system)

    def browse_structure(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.import_page.inputs_dialog,
            "Select atomistic LAMMPS structure",
            "",
            "LAMMPS data files (*.data *.dat);;All files (*)",
        )
        if path:
            self.import_page.structure_file.setText(path)
            self._commit_current_system()
            if self.project is not None and self.current_system_id is not None:
                system = self.project.system(self.current_system_id)
                system.structure_inferred = False
                system.infer_bonds_from_trajectory = False
                self.project.save()

    def _validation_sampling_changed(self, settings):
        if self.project is None or self.current_system_id is None or self.runner.running:
            return
        system = self.project.system(self.current_system_id)
        settings = normalize_sampling(settings)
        if settings == normalize_sampling(system.validation_sampling):
            return
        system.validation_sampling = settings
        self._reset_from(system, 'validate')
        self.project.save()
        self._refresh_pages()
        self.statusBar().showMessage('Sampling saved. Run Compare AA and CG to update validation.', 6000)

    def browse_aa_properties(self):
        if self.project is None or self.current_system_id is None:
            return
        requests = self.validation_page.viewer.cg_property_requests()
        if not requests:
            self._message(
                "CG properties are not available",
                "Complete optimization and run validation before loading AA properties.",
            )
            return
        current = self.project.system(self.current_system_id).aa_properties_file
        initial = str(Path(current).parent) if current else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load atomistic properties",
            initial,
            "LAMMPS averaged output (*.out *.dat *.txt);;All files (*)",
        )
        if not path:
            return
        try:
            matched = load_property_statistics(path, requests)
        except (OSError, ValueError) as exc:
            self._message("Could not load AA properties", str(exc), critical=True)
            return
        if not matched:
            expected = ", ".join(item["column"] for item in requests)
            self._message(
                "No matching AA properties",
                "The selected file contains none of the properties reported by the CG result.\n\n"
                f"Expected one or more of: {expected}",
                critical=True,
            )
            return
        system = self.project.system(self.current_system_id)
        system.aa_properties_file = path
        self.project.save()
        self.validation_page.viewer.set_aa_properties_file(path)
        self.statusBar().showMessage(
            f"Loaded {len(matched)} matching AA properties from {Path(path).name}",
            6000,
        )

    def _prepare_inspection_source(self, system) -> bool:
        if system.structure_file and Path(system.structure_file).is_file():
            return True
        if not system.trajectory or not Path(system.trajectory).is_file():
            raise FileNotFoundError("Select an existing structure.data file or atomistic trajectory.")
        if system.infer_bonds_from_trajectory:
            return True
        answer = QtWidgets.QMessageBox.question(
            self.import_page.inputs_dialog if self.import_page.inputs_dialog.isVisible() else self,
            "Infer bonds from trajectory?",
            "No structure.data file was found. Use RDKit to infer molecular connectivity "
            "from positions in the first trajectory frame?\n\n"
            "This is less reliable than an explicit topology, especially without molecule IDs "
            "or image flags. The inferred structure will be saved in the project for inspection. "
            "The current workflow still requires a LAMMPS-style dump containing id, coordinates, "
            "and element or mass columns.",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        system.infer_bonds_from_trajectory = True
        system.structure_file = ""
        system.structure_inferred = False
        self.project.save()
        self.statusBar().showMessage("Stage 1 will infer and retain molecular connectivity", 6000)
        return True

    def _refresh_species_selector(self, system):
        previous_id = self.species_selector.currentData()
        self.species_selector.blockSignals(True)
        self.species_selector.clear()
        for species in system.detected_species:
            self.species_selector.addItem(
                f"{species.get('name', 'species')}  ·  {species.get('formula', '')}  ·  ×{species.get('count', 0)}",
                species.get("id"),
            )
        if previous_id:
            index = self.species_selector.findData(previous_id)
            self.species_selector.setCurrentIndex(index if index >= 0 else 0)
        elif self.species_selector.count():
            self.species_selector.setCurrentIndex(0)
        self.species_selector.blockSignals(False)
        self._render_selected_species()

    def _species_selection_changed(self, _index):
        self._render_selected_species()
        if not getattr(self, '_syncing_focus', False) and self.project and self.current_system_id:
            species = self.project.system(self.current_system_id).detected_species
            if 0 <= _index < len(species):
                self._focus_molecule(species[_index]['id'], 0)

    def _workflow_tab_changed(self, _index):
        self._place_species_preview()
        if (
            self.project is not None
            and self.current_system_id is not None
            and not self.runner.running
        ):
            try:
                self._commit_current_system()
                system = self.project.system(self.current_system_id)
                self.mapping_page.load_system(system)
                self.force_page.load_system(system, self.project.system_root(system))
                self.lammps_page.load_system_root(
                    self.project.system_root(system),
                    self.project.execution.lammps_command,
                )
                self._refresh_species_selector(system)
            except Exception as exc:
                self._append_log(f"Could not synchronize workflow pages: {exc}\n")
        self._render_selected_species()

    def preview_species(self, identity: str):
        if not identity or self.project is None or self.current_system_id is None:
            return
        system = self.project.system(self.current_system_id)
        for index, species in enumerate(system.detected_species):
            if identity in {species.get("id"), species.get("name"), species.get("smiles")}:
                self.species_selector.setCurrentIndex(index)
                return

    def _render_selected_species(self):
        self.mapping_state_badge.hide()
        if self.project is None or self.current_system_id is None:
            self.canvas.clear_all()
            self.mapping_state_badge.hide()
            self.canvas_summary.setText('Run Stage 1 to inspect molecular species')
            if hasattr(self, "mapping_status_panel"):
                self.mapping_status_panel.hide()
            return
        system = self.project.system(self.current_system_id)
        index = self.species_selector.currentIndex()
        if index < 0 or index >= len(system.detected_species):
            self.canvas.clear_all()
            self.mapping_state_badge.hide()
            self.canvas_mode_label.setText("Molecular species")
            self.canvas_summary.setText("Run Stage 1 to inspect molecular species")
            if hasattr(self, "mapping_status_panel"):
                self.mapping_status_panel.hide()
            return
        species = system.detected_species[index]
        smiles = species.get("smiles", "")
        mapping_view = self.tabs.currentIndex() == 1
        inspect_view = self.tabs.currentIndex() == 0
        manual_mapping = mapping_view and self.mapping_page.mapping_mode() == "manual_custom"
        aa_view = self.tabs.currentIndex() == 0 or manual_mapping
        self._loading_canvas = True
        try:
            self.canvas.set_bead_summary_mode(manual_mapping)
            self.canvas.set_atom_circle_mode(inspect_view)
            if manual_mapping:
                creating = self.mapping_page.bead_creation_active()
                self.canvas.set_selection_only(creating)
                if not creating:
                    self.canvas.set_read_only(True)
                self.canvas.set_united_atom_mode(False)
            else:
                self.canvas.set_selection_only(False)
                self.canvas.set_read_only(True)
                self.canvas.set_united_atom_mode(mapping_view or not aa_view)
            self.canvas.show_h = aa_view
            self.canvas.show_carbon_dots = aa_view
            self.canvas.load_smiles(smiles, explicit_hydrogens=aa_view)
            if manual_mapping:
                self.canvas_mode_label.setText("Progressive CG bead preview")
            elif mapping_view:
                self.canvas_mode_label.setText("United-atom mapping preview")
            else:
                self.canvas_mode_label.setText(
                    "All-atom molecular species" if aa_view else "United-atom mapping preview"
                )
            count = int(species.get("count", 0))
            if manual_mapping:
                coverage = self._mapping_coverage(system, species)
                bead_instances = sum(
                    len(rule.get("matches", []))
                    for rule in system.custom_mapping_rules
                    if rule.get("species_id") == species.get("id", "")
                )
                representation = (
                    f"{bead_instances} CG bead(s) · "
                    f"{coverage['unmapped']} unmapped atom(s) remain"
                )
            elif mapping_view:
                representation = f"{species.get('heavy_atom_count', '-')} UA beads per molecule"
            elif aa_view:
                representation = f"{species.get('atom_count', '-')} AA atoms per molecule"
            else:
                representation = f"{species.get('heavy_atom_count', '-')} UA beads per molecule"
            self.canvas_summary.setText(
                f"{species.get('name', 'species')}  ·  {species.get('formula', '')}  ·  ×{count}\n"
                f"{representation}  ·  {smiles}"
            )
            self.canvas_species_button.setToolTip(self.species_selector.currentText())
            self.canvas_view_button.setToolTip(self.canvas_summary.text())
            self._update_bead_overlays(system, species)
        except Exception as exc:
            self.canvas.clear_all()
            self._append_log(f"Could not preview detected species: {exc}\n")
        finally:
            self._loading_canvas = False
            QtCore.QTimer.singleShot(0, self._fit_species_canvas)
            if manual_mapping:
                if self.mapping_page.bead_creation_active():
                    self._preview_custom_mapping_rule()
                else:
                    self.mapping_page.set_custom_rule_preview(None)
                    try:
                        self._update_mapping_status(system, species)
                    except Exception:
                        pass
            elif hasattr(self.canvas, "set_bead_overlays"):
                self.canvas.set_bead_overlays([])

    def _canvas_changed(self):
        if self._loading_canvas:
            return
        if hasattr(self, "tabs") and self.tabs.currentIndex() == 1:
            if self.mapping_page.mapping_mode() == "manual_custom" and self.mapping_page.bead_creation_active():
                self._preview_custom_mapping_rule()
            else:
                try:
                    system, species = self._selected_system_and_species()
                    self._update_mapping_status(system, species)
                except Exception:
                    pass

    def _selected_system_and_species(self):
        if self.project is None or self.current_system_id is None:
            raise ValueError("Open a project and select a system first.")
        system = self.project.system(self.current_system_id)
        index = self.species_selector.currentIndex()
        if index < 0 or index >= len(system.detected_species):
            raise ValueError("Select a detected species first.")
        return system, system.detected_species[index]

    def _open_custom_mapping_dialog(self):
        try:
            if self.project is None:
                raise ValueError("Open a project first.")
            if self.mapping_page.mapping_mode() != "manual_custom":
                raise ValueError("Switch Stage 2 to Manual custom beads first.")
            system, species = self._selected_system_and_species()
            suggested = f"B{len(system.custom_mapping_rules) + 1}"
            dialog = BeadRuleDialog(species, suggested, self)
            self._bead_dialog = dialog
            coverage = self._mapping_coverage(system, species)
            dialog.canvas.set_locked_atoms(coverage["mapped_atoms"])
            dialog.canvas.set_bead_overlays(self._bead_overlays_for(system, species))
            dialog.canvas.changed.connect(self._preview_bead_dialog)
            dialog.previewRequested.connect(self._preview_bead_dialog)
            dialog.createRequested.connect(self._save_bead_dialog)
            dialog.set_preview(None)
            dialog.exec()
        except Exception as exc:
            self._message("Could not open bead editor", str(exc), critical=True)
        finally:
            self._bead_dialog = None
            if self.project is not None and self.current_system_id is not None:
                system = self.project.system(self.current_system_id)
                self.mapping_page.load_system(system)
                self._render_selected_species()
                self._sync_visuals()

    def _preview_bead_dialog(self):
        dialog = self._bead_dialog
        if dialog is None:
            return
        try:
            system, species = self._selected_system_and_species()
            preview = self._build_custom_mapping_rule_preview(editor=dialog)
        except Exception as exc:
            preview = {"error": str(exc)}
        dialog.set_preview(preview)
        try:
            active_preview = preview if not preview.get("error") else None
            dialog.canvas.set_bead_overlays(
                self._bead_overlays_for(system, species, active_preview)
            )
            dialog.canvas.set_locked_atoms(
                self._mapping_coverage(system, species)["mapped_atoms"]
            )
        except Exception:
            pass

    def _save_bead_dialog(self):
        dialog = self._bead_dialog
        if dialog is None or self.project is None:
            return
        try:
            system, species = self._selected_system_and_species()
            preview = self._build_custom_mapping_rule_preview(
                require_label=True, editor=dialog
            )
            if not preview.get("matches"):
                raise ValueError(
                    "The selected bead signature produced no matching bead instances."
                )
            self._reject_rule_overlap(system, species, preview)
            label = preview["bead_label"]
            species_id = preview["species_id"]
            system.custom_mapping_rules = [
                rule
                for rule in system.custom_mapping_rules
                if not (
                    rule.get("bead_label") == label
                    and rule.get("species_id") == species_id
                )
            ]
            system.custom_mapping_rules.append(preview)
            self._reset_from(system, "cgmodel")
            self.project.save()
            dialog.accept()
            self.statusBar().showMessage(f"Saved custom bead rule {label}", 4000)
        except Exception as exc:
            self._message("Could not save custom bead rule", str(exc), critical=True)

    def _preview_custom_mapping_rule(self):
        if self.project is None or self.current_system_id is None:
            return
        if not self.mapping_page.bead_creation_active():
            self.mapping_page.set_custom_rule_preview(None)
            try:
                system, species = self._selected_system_and_species()
                self._update_bead_overlays(system, species, preview=None)
                self._update_mapping_status(system, species, preview=None)
            except Exception:
                pass
            return
        if self.tabs.currentIndex() != 1 or self.mapping_page.mapping_mode() != "manual_custom":
            self.mapping_page.set_custom_rule_preview(None)
            if hasattr(self.canvas, "set_bead_overlays"):
                self.canvas.set_bead_overlays([])
            return
        try:
            preview = self._build_custom_mapping_rule_preview()
        except Exception as exc:
            preview = {"error": str(exc)}
        self.mapping_page.set_custom_rule_preview(preview)
        try:
            system, species = self._selected_system_and_species()
            active_preview = preview if not preview.get("error") else None
            self._update_bead_overlays(system, species, preview=active_preview)
            self._update_mapping_status(system, species, preview=active_preview)
        except Exception:
            pass

    def _save_custom_mapping_rule(self):
        if self.project is None:
            return
        try:
            if self.mapping_page.mapping_mode() != "manual_custom":
                raise ValueError("Switch Stage 2 to Manual custom beads before saving a custom rule.")
            if not self.mapping_page.bead_creation_active():
                raise ValueError("Click Create new bead before selecting and finishing a bead type.")
            system, species = self._selected_system_and_species()
            preview = self._build_custom_mapping_rule_preview(require_label=True)
            if preview.get("error"):
                raise ValueError(str(preview["error"]))
            if not preview.get("matches"):
                raise ValueError("The selected bead signature produced no matching bead instances.")
            self._reject_rule_overlap(system, species, preview)
            label = preview["bead_label"]
            species_id = preview["species_id"]
            system.custom_mapping_rules = [
                rule
                for rule in system.custom_mapping_rules
                if not (rule.get("bead_label") == label and rule.get("species_id") == species_id)
            ]
            system.custom_mapping_rules.append(preview)
            self._reset_from(system, "cgmodel")
            self.project.save()
            self.canvas.clear_selection()
            self.mapping_page.set_bead_creation_active(False)
            self.mapping_page.load_system(system)
            self.mapping_page.set_custom_rule_preview(preview)
            self._update_bead_overlays(system, species, preview=None)
            self._update_mapping_status(system, species, preview=None)
            self.statusBar().showMessage(f"Saved custom bead rule {label}", 4000)
        except Exception as exc:
            self._message("Could not save custom bead rule", str(exc), critical=True)

    def _build_custom_mapping_rule_preview(
        self, require_label: bool = False, editor: BeadRuleDialog | None = None
    ) -> dict:
        system, species = self._selected_system_and_species()
        controls = editor if editor is not None else self.mapping_page
        selection_canvas = editor.canvas if editor is not None else self.canvas
        label = controls.bead_label()
        if require_label and not label:
            raise ValueError("Bead label is required.")
        if not label:
            label = f"B{len(system.custom_mapping_rules) + 1}"
        mol = self._selected_species_mol(species)
        selected_indices = sorted(
            aid
            for aid in selection_canvas.selected_atoms
            if 0 <= aid < mol.GetNumAtoms()
        )
        if len(selected_indices) != len(selection_canvas.selected_atoms):
            raise ValueError("Canvas atom selection does not match the RDKit molecule atom order.")
        if not selected_indices:
            raise ValueError("Select a connected atom group on the molecule canvas first.")
        coverage = self._mapping_coverage(system, species)
        blocked = sorted(set(selected_indices) & set(coverage["mapped_atoms"]))
        if blocked:
            raise ValueError(f"Selected atoms already belong to saved bead rules: {blocked}.")
        if not self._selection_connected(mol, selected_indices):
            raise ValueError("Selected atoms must form one connected bead group.")
        settings = controls.descriptor_settings()
        matches = self._find_subgraph_matches(mol, selected_indices, settings)
        descriptor = self._subgraph_descriptor(mol, selected_indices, settings)
        color = self._color_for_label(label)
        signature = ";".join(f"{key}={descriptor[key]}" for key in sorted(descriptor))
        return {
            "bead_label": label,
            "species_id": species.get("id", ""),
            "species_name": species.get("name", "species"),
            "species_smiles": species.get("smiles", ""),
            "prototype_atoms": selected_indices,
            "settings": settings,
            "descriptor": descriptor,
            "signature": signature,
            "color": color,
            "matches": matches,
        }

    def _selected_species_mol(self, species: dict):
        smiles = species.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Selected species has no valid SMILES for descriptor matching.")
        return Chem.AddHs(mol)

    def _selection_connected(self, mol, atom_indices: list[int]) -> bool:
        selected = set(atom_indices)
        seen = {atom_indices[0]}
        stack = [atom_indices[0]]
        while stack:
            current = stack.pop()
            atom = mol.GetAtomWithIdx(current)
            for neighbor in atom.GetNeighbors():
                idx = neighbor.GetIdx()
                if idx in selected and idx not in seen:
                    seen.add(idx)
                    stack.append(idx)
        return seen == selected

    def _find_subgraph_matches(self, mol, selected_indices: list[int], settings: dict) -> list[dict]:
        pattern = list(selected_indices)
        selected = set(pattern)
        pattern_descriptors = {idx: self._atom_descriptor(mol, idx, settings) for idx in pattern}
        pattern_edges = {}
        pattern_neighbors = {idx: set() for idx in pattern}
        for bond in mol.GetBonds():
            first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if first in selected and second in selected:
                key = tuple(sorted((first, second)))
                pattern_edges[key] = self._bond_descriptor(bond)
                pattern_neighbors[first].add(second)
                pattern_neighbors[second].add(first)
        candidates = {
            idx: [
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if self._atom_descriptor(mol, atom.GetIdx(), settings) == pattern_descriptors[idx]
            ]
            for idx in pattern
        }
        if any(not values for values in candidates.values()):
            return []
        order = sorted(pattern, key=lambda idx: (len(candidates[idx]), -len(pattern_neighbors[idx]), idx))
        matches: set[tuple[int, ...]] = set()
        mapping: dict[int, int] = {}
        used: set[int] = set()

        def compatible(pattern_idx: int, candidate_idx: int) -> bool:
            for other_pattern, other_candidate in mapping.items():
                expected = pattern_edges.get(tuple(sorted((pattern_idx, other_pattern))))
                actual_bond = mol.GetBondBetweenAtoms(candidate_idx, other_candidate)
                if expected is None:
                    if actual_bond is not None:
                        return False
                elif actual_bond is None or self._bond_descriptor(actual_bond) != expected:
                    return False
            return True

        def search(position: int):
            if position == len(order):
                atom_set = tuple(sorted(mapping.values()))
                matches.add(atom_set)
                return
            pattern_idx = order[position]
            for candidate_idx in candidates[pattern_idx]:
                if candidate_idx in used or not compatible(pattern_idx, candidate_idx):
                    continue
                mapping[pattern_idx] = candidate_idx
                used.add(candidate_idx)
                search(position + 1)
                used.remove(candidate_idx)
                del mapping[pattern_idx]

        search(0)
        result = []
        for atom_ids in sorted(matches):
            heavy = [idx for idx in atom_ids if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1]
            result.append({"center": heavy[0] if heavy else atom_ids[0], "atoms": list(atom_ids)})
        return result

    def _atom_descriptor(self, mol, atom_idx: int, settings: dict) -> tuple:
        atom = mol.GetAtomWithIdx(atom_idx)
        parts: list[object] = [("element", atom.GetSymbol())]
        if settings.get("match_hydrogens", True):
            parts.append(("hydrogens", sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1)))
        if settings.get("match_aromatic", True):
            parts.append(("aromatic", atom.GetIsAromatic()))
        if settings.get("match_ring", True):
            parts.append(("ring", atom.IsInRing()))
        if settings.get("match_hybridization", True):
            parts.append(("hybridization", str(atom.GetHybridization())))
        if settings.get("match_charge", True):
            parts.append(("formal_charge", atom.GetFormalCharge()))
        if settings.get("match_neighbors", True):
            depth = int(settings.get("neighbor_depth", 1))
            parts.append(("degree", atom.GetDegree()))
            for radius in range(1, depth + 1):
                parts.append((f"shell_{radius}", tuple(self._neighbor_shell(mol, atom_idx, radius))))
        return tuple(parts)

    def _bond_descriptor(self, bond) -> str:
        return str(bond.GetBondType())

    def _neighbor_shell(self, mol, atom_idx: int, radius: int) -> list[str]:
        visited = {atom_idx}
        frontier = {atom_idx}
        for _step in range(radius):
            next_frontier = set()
            for current in frontier:
                atom = mol.GetAtomWithIdx(current)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in visited:
                        continue
                    next_frontier.add(neighbor_idx)
            visited.update(next_frontier)
            frontier = next_frontier
        return sorted(mol.GetAtomWithIdx(idx).GetSymbol() for idx in frontier)

    def _subgraph_descriptor(self, mol, atom_indices: list[int], settings: dict) -> dict:
        atom_set = set(atom_indices)
        elements = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in atom_indices]
        counts = {element: elements.count(element) for element in sorted(set(elements))}
        bonds = []
        for bond in mol.GetBonds():
            first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if first in atom_set and second in atom_set:
                bonds.append(f"{mol.GetAtomWithIdx(first).GetSymbol()}-{self._bond_descriptor(bond)}-{mol.GetAtomWithIdx(second).GetSymbol()}")
        smarts = Chem.MolFragmentToSmarts(mol, atomsToUse=atom_indices)
        return {
            "atom_count": len(atom_indices),
            "heavy_count": sum(1 for idx in atom_indices if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1),
            "formula": self._formula_from_elements(elements),
            "elements": ",".join(f"{key}{counts[key]}" for key in sorted(counts)),
            "bonds": ",".join(sorted(bonds)),
            "smarts": smarts,
            "settings": settings,
        }

    def _formula_from_elements(self, elements: list[str]) -> str:
        counts = {element: elements.count(element) for element in set(elements)}
        if "C" in counts:
            order = ["C"]
            if "H" in counts:
                order.append("H")
            order.extend(sorted(element for element in counts if element not in {"C", "H"}))
        else:
            order = sorted(counts)
        return "".join(element + (str(counts[element]) if counts[element] != 1 else "") for element in order)

    def _color_for_label(self, label: str) -> str:
        return bead_color(label)

    def _bead_creation_mode_changed(self, active: bool):
        if hasattr(self, "canvas"):
            self.canvas.clear_selection()
        self.mapping_page.set_custom_rule_preview(None)
        self._render_selected_species()

    def _cancel_custom_mapping_rule(self):
        if hasattr(self, "canvas"):
            self.canvas.clear_selection()
        self.mapping_page.set_custom_rule_preview(None)
        try:
            system, species = self._selected_system_and_species()
            self._update_bead_overlays(system, species)
            self._update_mapping_status(system, species)
        except Exception:
            pass

    def _clear_mapping_selection(self):
        if hasattr(self, "canvas"):
            self.canvas.clear_selection()
        try:
            system, species = self._selected_system_and_species()
            self._update_bead_overlays(system, species)
            self._update_mapping_status(system, species)
        except Exception:
            pass

    def _delete_selected_custom_rule(self, rule_index: int):
        if self.project is None or self.current_system_id is None:
            return
        try:
            system = self.project.system(self.current_system_id)
            if rule_index < 0 or rule_index >= len(system.custom_mapping_rules):
                raise IndexError("The selected bead rule is no longer available.")
            removed = system.custom_mapping_rules.pop(rule_index)
            self._reset_from(system, "cgmodel")
            self.canvas.clear_selection()
            self.project.save()
            self.mapping_page.load_system(system)
            self.mapping_page.set_custom_rule_preview(None)
            self._render_selected_species()
            self._sync_visuals()
            label = removed.get("bead_label", "bead")
            species_name = removed.get("species_name", "species")
            self.statusBar().showMessage(
                f"Deleted bead rule {label} from {species_name}", 4000
            )
        except Exception as exc:
            self._message("Could not delete custom bead rule", str(exc), critical=True)

    def _clear_current_species_rules(self):
        if self.project is None:
            return
        try:
            system, species = self._selected_system_and_species()
            species_id = species.get("id", "")
            system.custom_mapping_rules = [
                rule for rule in system.custom_mapping_rules if rule.get("species_id") != species_id
            ]
            self._reset_from(system, "cgmodel")
            self.canvas.clear_selection()
            self.project.save()
            self.mapping_page.load_system(system)
            self.mapping_page.set_custom_rule_preview(None)
            self._update_bead_overlays(system, species)
            self._update_mapping_status(system, species)
            self.statusBar().showMessage(f"Cleared custom bead rules for {species.get('name', 'species')}", 4000)
            self._sync_visuals()
        except Exception as exc:
            self._message("Could not clear custom bead rules", str(exc), critical=True)

    def _mapping_coverage(self, system, species: dict) -> dict:
        mol = self._selected_species_mol(species)
        total = mol.GetNumAtoms()
        counts = {index: 0 for index in range(total)}
        species_id = species.get("id", "")
        rule_count = 0
        for rule in system.custom_mapping_rules:
            if rule.get("species_id") != species_id:
                continue
            rule_count += 1
            for match in rule.get("matches", []):
                for atom_id in match.get("atoms", []):
                    atom_id = int(atom_id)
                    if 0 <= atom_id < total:
                        counts[atom_id] += 1
        mapped_atoms = sorted(atom_id for atom_id, count in counts.items() if count > 0)
        overlap_atoms = sorted(atom_id for atom_id, count in counts.items() if count > 1)
        unmapped_atoms = sorted(atom_id for atom_id, count in counts.items() if count == 0)
        return {
            "total": total,
            "mapped": len(mapped_atoms),
            "unmapped": len(unmapped_atoms),
            "overlaps": len(overlap_atoms),
            "mapped_atoms": mapped_atoms,
            "overlap_atoms": overlap_atoms,
            "unmapped_atoms": unmapped_atoms,
            "rule_count": rule_count,
            "complete": len(mapped_atoms) == total and not overlap_atoms and rule_count > 0,
        }

    def _reject_rule_overlap(self, system, species: dict, preview: dict):
        coverage = self._mapping_coverage(system, species)
        reserved = set(coverage["mapped_atoms"])
        overlapping = sorted(
            atom_id
            for match in preview.get("matches", [])
            for atom_id in match.get("atoms", [])
            if int(atom_id) in reserved
        )
        if overlapping:
            raise ValueError(
                "Detected equivalent beads overlap saved bead rules. "
                f"Overlapping atom indices: {overlapping}."
            )

    def _all_manual_species_mapped(self, system) -> tuple[bool, list[str]]:
        if system.mapping_mode != "manual_custom":
            return True, []
        problems = []
        for species in system.detected_species:
            coverage = self._mapping_coverage(system, species)
            name = species.get("name", species.get("id", "species"))
            if coverage["overlaps"]:
                problems.append(f"{name}: {coverage['overlaps']} overlapping atom(s)")
            if coverage["unmapped"]:
                problems.append(f"{name}: {coverage['unmapped']} unmapped atom(s)")
            if coverage["rule_count"] == 0:
                problems.append(f"{name}: no bead rules")
        return not problems, problems

    def _update_mapping_status(self, system, species: dict, preview: dict | None = None):
        if not hasattr(self, "mapping_status_panel"):
            return
        mapping_tab = hasattr(self, "tabs") and self.tabs.currentIndex() == 1
        manual = mapping_tab and self.mapping_page.mapping_mode() == "manual_custom"
        self.mapping_state_badge.setVisible(manual)
        if not manual:
            return
        coverage = self._mapping_coverage(system, species)
        selected = len(getattr(self.canvas, "selected_atoms", []))
        preview_matches = len(preview.get("matches", [])) if preview else 0
        self.mapping_selected_label.setText(f"Selected: {selected}")
        self.mapping_matches_label.setText(f"Matches: {preview_matches}")
        self.mapping_saved_label.setText(f"Types: {coverage['rule_count']}")
        self.mapping_mapped_label.setText(f"Mapped: {coverage['mapped']}/{coverage['total']}")
        self.mapping_unmapped_label.setText(f"Unmapped: {coverage['unmapped']}")
        self.mapping_status_panel.setToolTip(
            f"Unmapped atoms: {coverage['unmapped']} · Overlapping atoms: {coverage['overlaps']}"
        )
        self.mapping_state_badge.setToolTip(
            f"{self.mapping_saved_label.text()} · {self.mapping_mapped_label.text()}\n"
            + self.mapping_status_panel.toolTip()
        )
        if coverage["complete"]:
            self.mapping_state_badge.setText("Fully mapped")
            self.mapping_state_badge.setProperty("state", "complete")
        elif coverage["overlaps"]:
            self.mapping_state_badge.setText(f"Overlaps: {coverage['overlaps']}")
            self.mapping_state_badge.setProperty("state", "failed")
        else:
            self.mapping_state_badge.setText("Incomplete")
            self.mapping_state_badge.setProperty("state", "pending")
        self.mapping_state_badge.style().unpolish(self.mapping_state_badge)
        self.mapping_state_badge.style().polish(self.mapping_state_badge)

    def _update_bead_overlays(self, system, species: dict, preview: dict | None = None):
        if not hasattr(self.canvas, "set_bead_overlays"):
            return
        if self.tabs.currentIndex() != 1 or self.mapping_page.mapping_mode() != "manual_custom":
            self.canvas.set_bead_overlays([])
            if hasattr(self.canvas, "set_locked_atoms"):
                self.canvas.set_locked_atoms([])
            return
        self.canvas.set_bead_overlays(
            self._bead_overlays_for(system, species, preview)
        )
        if hasattr(self.canvas, "set_locked_atoms"):
            self.canvas.set_locked_atoms(self._mapping_coverage(system, species)["mapped_atoms"])

    def _bead_overlays_for(self, system, species: dict, preview: dict | None = None):
        overlays = []
        species_id = species.get("id", "")
        for rule in system.custom_mapping_rules:
            if rule.get("species_id") != species_id:
                continue
            color = rule.get("color") or self._color_for_label(rule.get("bead_label", ""))
            for match in rule.get("matches", []):
                overlays.append(
                    {
                        "atom_ids": list(match.get("atoms", [])),
                        "color": color,
                        "label": rule.get("bead_label", ""),
                        "active": False,
                    }
                )
        if preview:
            color = preview.get("color") or self._color_for_label(preview.get("bead_label", ""))
            for match in preview.get("matches", []):
                overlays.append(
                    {
                        "atom_ids": list(match.get("atoms", [])),
                        "color": color,
                        "label": preview.get("bead_label", ""),
                        "active": True,
                    }
                )
        return overlays

    def _prepare_process_dialog(self, stage: str, system_id: str | None):
        if self._process_dialog is not None and not self._process_dialog.running:
            self._process_dialog.close()
        if self.project is None:
            raise RuntimeError("Open a project before starting a workflow stage.")
        system_name = (
            "Selected systems"
            if stage == "merge"
            else self.project.system(system_id).name
        )
        dialog = ProcessDialog(stage, system_name, self.project.path, self)
        dialog.cancelRequested.connect(self.runner.cancel)
        dialog.finished.connect(self._process_dialog_finished)
        self._process_dialog = dialog
        self._log_buffer = ""
        self._append_log(f"Project: {self.project.path}\n")
        return dialog

    def _process_dialog_finished(self, _result: int):
        dialog = self.sender()
        if dialog is self._process_dialog:
            self._process_dialog = None

    def run_stage(self, stage: str, continuation=None):
        if self.project is None:
            return
        if self.runner.running:
            self._message("Workflow busy", "Wait for or cancel the current stage first.")
            return
        try:
            self._commit_current_system()
            if stage == "merge":
                if len(self.project.merge_system_ids) < 2:
                    raise ValueError("Select at least two systems in the Merge tab.")
                self.project.merge_status = "running"
                for selected_id in self.project.merge_system_ids:
                    self.project.system(selected_id).stages["validate"] = "pending"
                system_id = None
            else:
                if self.current_system_id is None:
                    raise ValueError("Select a system first.")
                system_id = self.current_system_id
                system = self.project.system(system_id)
                if stage == "inspect" and not self._prepare_inspection_source(system):
                    return
                self._validate_stage(system, stage)
                if system.stages.get(stage) == "complete" and continuation is None:
                    answer = QtWidgets.QMessageBox.question(
                        self,
                        "Rerun stage",
                        ("Start a new independent optimization study? Previous studies stay intact. "
                         "Use Continue… to add trials to the existing study." if stage == 'optimize' else
                         "This stage is already complete. Rerunning it may replace its generated outputs. Continue?"),
                    )
                    if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                        return
                self._reset_from(system, stage)
                system.stages[stage] = "running"
            self.project.save()
            self._refresh_pages()
            process_dialog = self._prepare_process_dialog(stage, system_id)
            if continuation is None:
                self.runner.start(self.project.path, stage, system_id)
            else:
                self.runner.start(self.project.path, stage, system_id, continuation=continuation)
            process_dialog.show()
            process_dialog.raise_()
            process_dialog.activateWindow()
            self.tabs.setEnabled(False)
            self.system_list.setEnabled(False)
            self.add_system_button.setEnabled(False)
            self.remove_system_button.setEnabled(False)
            self.statusBar().showMessage(f"Running {stage}…")
        except Exception as exc:
            if self._process_dialog is not None and not self.runner.running:
                self._process_dialog._running = False
                self._process_dialog.close()
            if stage == "merge":
                self.project.merge_status = "failed"
            elif self.current_system_id:
                self.project.system(self.current_system_id).stages[stage] = "failed"
            self.project.save()
            self._refresh_pages()
            self._message("Cannot start stage", str(exc), critical=True)

    def run_smoke_test(self):
        if self.project is None:
            return
        if self.runner.running:
            self._message("Workflow busy", "Wait for or cancel the current stage first.")
            return
        system_id = self.current_system_id
        try:
            self._commit_current_system()
            if system_id is None:
                raise ValueError("Select a system first.")
            system = self.project.system(system_id)
            root = self.project.system_root(system)
            required = [
                root / "output" / "lammps" / "input.data",
                root / "output" / "lammps" / "structure.data",
            ]
            missing = [path.name for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Generate the LAMMPS setup first. Missing: "
                    + ", ".join(missing)
                )
            if not self.project.execution.lammps_command:
                raise ValueError("Configure the command in the LAMMPS tab first.")
            if system.stages.get("smoke") == "complete":
                answer = QtWidgets.QMessageBox.question(
                    self,
                    "Rerun smoke test",
                    "The generated setup already passed its smoke test. Run it again?",
                )
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            system.stages["smoke"] = "running"
            self.project.save()
            self._refresh_pages()
            process_dialog = self._prepare_process_dialog("smoke", system_id)
            self.runner.start(self.project.path, "smoke", system_id)
            process_dialog.show()
            process_dialog.raise_()
            process_dialog.activateWindow()
            self.tabs.setEnabled(False)
            self.system_list.setEnabled(False)
            self.add_system_button.setEnabled(False)
            self.remove_system_button.setEnabled(False)
            self.statusBar().showMessage("Running LAMMPS smoke test…")
        except Exception as exc:
            if self._process_dialog is not None and not self.runner.running:
                self._process_dialog._running = False
                self._process_dialog.close()
            if system_id is not None:
                self.project.system(system_id).stages["smoke"] = "failed"
            self.project.save()
            self._refresh_pages()
            self._message("Cannot start smoke test", str(exc), critical=True)

    def _lammps_command_changed(self, text: str):
        if self.project is None:
            return
        command = split_command(text)
        if not command:
            self._message(
                "LAMMPS command required",
                "Enter the command used to launch LAMMPS.",
                critical=True,
            )
            self.lammps_page.command.setText(
                display_command(self.project.execution.lammps_command)
            )
            return
        self.project.execution.lammps_command = command
        self.optimize_page.command.setText(display_command(command))
        self.project.save()
        self.lammps_page._update_smoke_button()
        self.statusBar().showMessage("LAMMPS command updated", 3000)

    def _validate_stage(self, system, stage: str):
        if stage == "inspect":
            if not system.trajectory or not Path(system.trajectory).is_file():
                raise FileNotFoundError("Select an existing atomistic trajectory.")
            if not (
                system.structure_file and Path(system.structure_file).is_file()
            ) and not system.infer_bonds_from_trajectory:
                raise FileNotFoundError("Select a structure file or approve bond inference.")
            return
        if stage == "cgmodel":
            if not system.structure_file or not Path(system.structure_file).is_file():
                raise FileNotFoundError("Select an existing atomistic structure.data file.")
            if not system.detected_species:
                raise ValueError("Complete Stage 1 before building the CG model.")
            if not system.trajectory or not Path(system.trajectory).is_file():
                raise FileNotFoundError("Select an existing atomistic trajectory.")
            if system.stages.get("inspect") != "complete":
                raise RuntimeError("Complete Stage 1 inspection first.")
            failed_checks = [
                check.get("name", "check")
                for check in system.trajectory_stats.get("checks", [])
                if check.get("status") == "fail"
            ]
            if failed_checks:
                raise RuntimeError(
                    "Resolve the failed Stage 1 checks before building the CG model: "
                    + ", ".join(failed_checks)
                )
            complete, problems = self._all_manual_species_mapped(system)
            if not complete:
                raise RuntimeError(
                    "Complete the manual bead mapping before building the CG model: "
                    + "; ".join(problems)
                )
            return
        prerequisites = {
            "force": "cgmodel",
            "postprocess": "force",
            "lammps": "postprocess",
            "optimize": "lammps",
            "validate": "lammps",
        }
        previous = prerequisites[stage]
        stage_names = {
            "cgmodel": "Stage 2 CG model",
            "force": "Fit",
            "postprocess": "Prepare tables",
            "lammps": "LAMMPS preflight",
            "optimize": "Refine / Property optimization",
        }
        if system.stages.get(previous) != "complete":
            raise RuntimeError(f"Complete {stage_names.get(previous, previous)} first.")
        if stage == "validate" and self.project is not None:
            if self.project.validation_trajectory(system) is None:
                raise FileNotFoundError(
                    "Validate needs a CG simulation trajectory. Run optional Refine, or run "
                    "the generated LAMMPS setup externally to produce output/lammps/cg.lammpstrj. "
                    "The smoke test does not generate a trajectory."
                )
        if stage == "force" and self.project is not None:
            output = self.project.system_root(system) / "output"
            required = [
                output / "work" / "top.in",
                output / "work" / "types_cg.lammpstrj",
                output / "work" / "fit_plan.json",
            ]
            missing = [path.name for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError("Stage 2 CG model outputs are missing: " + ", ".join(missing))
        if stage == "optimize" and not system.optimization_properties:
            raise ValueError("Add at least one property to the Optuna objective.")
        if stage == "optimize" and system.optimization_protocol and self.project is not None:
            from .simulation_protocol import prepare_trial_input, read_protocol, validate_scoring_window
            source = self.project.system_root(system) / 'output' / 'lammps' / 'input.data'
            prepared = prepare_trial_input(source.read_text(encoding='utf-8'), system.optimization_protocol)
            validate_scoring_window(read_protocol(prepared)[0], system.optimization_properties)

    def _reset_from(self, system, stage: str):
        index = STAGE_ORDER.index(stage)
        for downstream in STAGE_ORDER[index:]:
            system.stages[downstream] = "pending"
            if downstream not in system.invalidated_stages:
                system.invalidated_stages.append(downstream)
        if index <= STAGE_ORDER.index("lammps"):
            system.stages["smoke"] = "pending"
            if 'smoke' not in system.invalidated_stages:
                system.invalidated_stages.append('smoke')
        if self.project and stage != "validate":
            self.project.merge_status = "pending"
            self.project.merge_invalidated = True

    def _stage_finished(self, stage: str, system_id, success: bool):
        if self.project is None:
            return
        if stage in {"inspect", "detect"}:
            current_id = self.current_system_id
            self.project = Project.load(self.project.path)
            self.current_system_id = current_id
            if system_id:
                self.project.system(system_id).stages["inspect"] = (
                    "complete" if success else "failed"
                )
                if success:
                    system = self.project.system(system_id)
                    system.invalidated_stages = [s for s in system.invalidated_stages if s != 'inspect']
                self.project.save()
            if current_id:
                self._load_system(self.project.system(current_id))
            self.tabs.setEnabled(True)
            self.system_list.setEnabled(True)
            self.add_system_button.setEnabled(True)
            self.remove_system_button.setEnabled(True)
            self._refresh_system_list()
            self._refresh_pages()
            result = "completed" if success else "failed"
            self.statusBar().showMessage(f"Stage 1 inspection {result}", 8000)
            self._append_log(
                "\n✓ Stage 1 inspection completed.\n"
                if success
                else "\n✕ Stage 1 inspection failed. See the error above.\n"
            )
            if self._process_dialog is not None:
                self._process_dialog.finish_run(success)
            return
        if stage == "merge":
            self.project.merge_status = "complete" if success else "failed"
            if success:
                self.project.merge_invalidated = False
        elif system_id:
            self.project.system(system_id).stages[stage] = "complete" if success else "failed"
            if success:
                system = self.project.system(system_id)
                system.invalidated_stages = [s for s in system.invalidated_stages if s != stage]
        self.project.save()
        self.tabs.setEnabled(True)
        self.system_list.setEnabled(True)
        self.add_system_button.setEnabled(True)
        self.remove_system_button.setEnabled(True)
        self._refresh_system_list()
        self._refresh_pages()
        if stage == "postprocess" and success:
            self.post_page.show_accepted()
        result = "completed" if success else "failed"
        stage_name = STAGE_LABELS.get(stage, stage.title())
        self.statusBar().showMessage(f"{stage_name} {result}", 8000)
        if success:
            self._append_log(f"\n✓ {stage_name} completed successfully.\n")
        else:
            self._append_log(f"\n✕ {stage_name} failed. See the error above.\n")
        if self._process_dialog is not None:
            self._process_dialog.finish_run(success)

    def _refresh_pages(self):
        if self.project is None:
            return
        if self.current_system_id:
            system = self.project.system(self.current_system_id)
            self.import_page.set_status(system.stages.get("inspect", "pending"), self.runner.running)
            self.import_page.update_input_summary(system)
            self.mapping_page.set_status(system.stages.get("cgmodel", "pending"), self.runner.running)
            root = self.project.system_root(system)
            self.force_page.load_system(system, root)
            self.force_page.set_status(system.stages.get("force", "pending"), self.runner.running)
            self.post_page.set_status(system.stages.get("postprocess", "pending"), self.runner.running)
            self.lammps_page.set_status(system.stages.get("lammps", "pending"), self.runner.running)
            self.lammps_page.set_smoke_status(
                system.stages.get("smoke", "pending"), self.runner.running
            )
            self.optimize_page.set_status(system.stages.get("optimize", "pending"), self.runner.running)
            self.optimize_page._refresh_protocol_summary()
            self.validation_page.set_status(system.stages.get("validate", "pending"), self.runner.running)
            self.post_page.load_system_root(root, system.table_overrides, system.stages.get('postprocess') == 'complete')
            self.lammps_page.load_system_root(root, self.project.execution.lammps_command)
            self.validation_page.load_system(system, root)
            self._sync_visuals()
        self.merge_page.load_project(self.project)
        self.merge_page.set_status(self.project.merge_status, self.runner.running)

    def _hydrate_statuses(self, project: Project):
        for system in project.systems:
            output = project.system_root(system) / "output"
            checks = {
                "inspect": bool(system.detected_species) and bool(system.trajectory_stats),
                "cgmodel": (
                    (output / "work" / "top.in").is_file()
                    and (output / "work" / "types_cg.lammpstrj").is_file()
                    and (output / "work" / "fit_plan.json").is_file()
                    and bool(list((output / "histograms").glob("histogram_xy_*.csv")))
                ),
                "force": bool(list((output / "tables").glob("*.table"))) and bool(list((output / "models").glob("*.p"))),
                "postprocess": bool(list((output / "processed_tables" / "2_accepted").glob("*.table"))),
                "lammps": (output / "lammps" / "input.data").is_file(),
                "smoke": (
                    (project.system_root(system) / "smoke_test" / "smoke-test.ok").is_file()
                    and system.stages.get("smoke") == "complete"
                ),
                "optimize": (best_directory(project.system_root(system) / 'optimization') / 'input.data').is_file(),
                "validate": (
                    (project.system_root(system) / "validation" / "metrics.json").is_file()
                    and system.stages.get("validate") != "pending"
                ),
            }
            for stage, complete in checks.items():
                if stage in system.invalidated_stages:
                    if system.stages.get(stage) == 'running':
                        system.stages[stage] = 'failed'
                    continue
                if complete:
                    system.stages[stage] = "complete"
                elif system.stages.get(stage) == "running":
                    system.stages[stage] = "failed"
        manifest = project.root / "merge" / "combined_forcefield" / "deterministic" / "manifest.json"
        if manifest.is_file() and not project.merge_invalidated:
            project.merge_status = "complete"
        elif project.merge_status == "running":
            project.merge_status = "failed"
        project.save()

    def open_project_folder(self):
        if self.project:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.project.root)))

    def _append_log(self, text: str):
        self._log_buffer = (self._log_buffer + text)[-1_000_000:]
        if self._process_dialog is not None:
            self._process_dialog.append_log(text)

    def _message(self, title: str, text: str, critical: bool = False):
        method = QtWidgets.QMessageBox.critical if critical else QtWidgets.QMessageBox.information
        method(self, title, text)

    def closeEvent(self, event):
        if self._export_dialog is not None and self._export_dialog.busy:
            self._export_dialog.reject()
            self.statusBar().showMessage('Cancelling export; close again once it finishes.', 6000)
            event.ignore()
            return
        if self.runner.running:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Stage still running",
                "Cancel the running stage and close the application?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.cancel()
        try:
            self._commit_current_system()
        except Exception:
            pass
        self.tutorial.stop()
        self.validation_page.comparison.shutdown()
        for view in self.trajectory_views[:2]:
            view.shutdown()
        event.accept()









