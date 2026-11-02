from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .molecule_drawer import MoleculeCanvas
from .optimization import PROPERTY_CATALOG, property_definition, property_settings
from .project import SystemSpec, display_command, split_command
from .resources import available_cores
from .results_viewers import TableExplorer, ValidationViewer
from .trajectory_view import TrajectoryView
from .trajectory_comparison import TrajectoryComparison
from .fit_review import FitReview
from .table_editor import TableProcessingExplorer
from .simulation_protocol import DEFAULT_PROTOCOL, protocol_summary, read_protocol, merge_protocol
from .simulation_protocol_dialog import SimulationProtocolDialog
from .validation_sampling import normalize_sampling
from .validation_sampling_dialog import ValidationSamplingDialog


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.3g} {unit}"
        value /= 1024.0
    return f"{value:.3g} TiB"


class StageDescription(QtWidgets.QLabel):
    """One-line subtitle; the full description remains available on hover."""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self._full_text = ''
        self.setMinimumWidth(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.setText(text)

    def setText(self, text):
        self._full_text = text
        self.setToolTip(text)
        super().setText(self.fontMetrics().elidedText(text, QtCore.Qt.TextElideMode.ElideRight, self.width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setText(self._full_text)


class PageBase(QtWidgets.QWidget):
    runRequested = QtCore.Signal(str)

    def __init__(
        self, title: str, description: str, parent=None, *, scrollable: bool = True
    ):
        super().__init__(parent)
        self.outer_layout = QtWidgets.QVBoxLayout(self)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.outer_layout.setSpacing(0)
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_content.setObjectName("workflowScrollContent")
        self.layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.layout.setContentsMargins(12, 8, 12, 8)
        if scrollable:
            self.scroll_area = QtWidgets.QScrollArea()
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.scroll_area.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
            self.scroll_area.setWidget(self.scroll_content)
            self.outer_layout.addWidget(self.scroll_area, 1)
        else:
            self.scroll_area = None
            self.layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetDefaultConstraint)
            self.outer_layout.addWidget(self.scroll_content, 1)
        self.header = QtWidgets.QWidget()
        self.header.setObjectName('stageHeader')
        self.header_layout = QtWidgets.QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(12, 6, 12, 6)
        self.header_layout.setSpacing(12)
        heading = QtWidgets.QLabel(title)
        self.heading = heading
        heading.setObjectName("pageTitle")
        body = StageDescription(description)
        self.description = body
        body.setObjectName("pageDescription")
        self.header_layout.addWidget(heading)
        self.header_layout.addWidget(body, 1)
        self.outer_layout.insertWidget(0, self.header)

    def enable_split_workspace(self, left_width: int = 600, right_width: int = 600):
        """Use a stage-local left/right workspace below the page heading."""
        page_layout = self.layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setObjectName("stageWorkspaceSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setMinimumHeight(260)

        preview = QtWidgets.QWidget()
        preview.setObjectName("stagePreviewPane")
        self.preview_layout = QtWidgets.QVBoxLayout(preview)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)

        controls = QtWidgets.QWidget()
        controls.setObjectName("stageControlsPane")
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetDefaultConstraint)

        splitter.addWidget(controls)
        splitter.addWidget(preview)
        splitter.setSizes([left_width, right_width])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        page_layout.addWidget(splitter, 1)

        self.workspace_splitter = splitter
        self.page_layout = page_layout
        self.layout = controls_layout

    def add_run_row(self, stage: str, output_label: str):
        self.status = QtWidgets.QLabel("Pending")
        self.status.setObjectName("stageStatus")
        self.output = QtWidgets.QLabel(output_label)
        self.output.setWordWrap(True)
        self.output.setObjectName("stageOutput")
        self.output.setMaximumHeight(42)
        self.output.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.action_text = {
            'inspect': 'Inspect inputs', 'cgmodel': 'Build mapped trajectory',
            'force': 'Fit force field', 'postprocess': 'Prepare tables',
            'lammps': 'Generate LAMMPS setup', 'optimize': 'Optimize force field',
            'validate': 'Compare AA and CG', 'merge': 'Combine systems',
        }.get(stage, 'Run stage')
        self.run_button = QtWidgets.QPushButton(self.action_text)
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(lambda: self.runRequested.emit(stage))
        self.run_bar = QtWidgets.QFrame()
        self.run_bar.setObjectName("stageActions")
        bar_layout = QtWidgets.QHBoxLayout(self.run_bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(6)
        bar_layout.addWidget(self.status)
        bar_layout.addWidget(self.run_button)
        self.output.hide()
        self.run_button.setToolTip(output_label)
        self.header_layout.addWidget(self.run_bar)

    def set_status(self, value: str, running: bool = False):
        labels = {
            "pending": "Pending",
            "running": "Running…",
            "complete": "Complete",
            "failed": "Failed",
        }
        self.status.setText(labels.get(value, value.title()))
        self.status.setProperty("state", value)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.run_button.setEnabled(not running)
        self.run_button.setText("Running…" if running else self.action_text)

    def add_subtabs(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("stageSubTabs")
        tabs.setDocumentMode(True)
        self.layout.addWidget(tabs, 1)
        return tabs

    def add_subtab(self, tabs: QtWidgets.QTabWidget, title: str) -> QtWidgets.QVBoxLayout:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        tabs.addTab(page, title)
        return layout


class ImportPage(PageBase):
    browseTrajectoryRequested = QtCore.Signal()
    browseStructureRequested = QtCore.Signal()
    speciesPreviewRequested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(
            "1 · Import and inspect",
            "Load the atomistic trajectory and topology, detect molecular species, and check whether the data are suitable for mapping.",
            parent,
            scrollable=False,
        )
        self.enable_split_workspace()
        self.inputs_dialog = QtWidgets.QDialog(self)
        self.inputs_dialog.setWindowTitle('System inputs')
        self.inputs_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.inputs_dialog.resize(700, 280)
        input_layout = QtWidgets.QVBoxLayout(self.inputs_dialog)
        self.details_dialog = QtWidgets.QDialog(self)
        self.details_dialog.setWindowTitle('Trajectory checks and detected species')
        self.details_dialog.resize(900, 650)
        details_layout = QtWidgets.QVBoxLayout(self.details_dialog)
        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName('stageSubTabs')
        tabs.setDocumentMode(True)
        details_layout.addWidget(tabs)
        summary_layout = self.add_subtab(tabs, "Trajectory")
        checks_layout = self.add_subtab(tabs, "Checks")
        species_layout_tab = self.add_subtab(tabs, "Species")

        form_box = QtWidgets.QGroupBox("System input")
        form = QtWidgets.QFormLayout(form_box)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("benzene")
        self.structure_file = QtWidgets.QLineEdit()
        self.structure_file.setPlaceholderText("Atomistic LAMMPS data file with Atoms and Bonds")
        structure_browse = QtWidgets.QPushButton("Browse…")
        structure_browse.clicked.connect(self.browseStructureRequested.emit)
        structure_row = QtWidgets.QHBoxLayout()
        structure_row.addWidget(self.structure_file, 1)
        structure_row.addWidget(structure_browse)
        self.trajectory = QtWidgets.QLineEdit()
        self.trajectory.setPlaceholderText("LAMMPS atomistic trajectory (*.lammpstrj)")
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self.browseTrajectoryRequested.emit)
        trajectory_row = QtWidgets.QHBoxLayout()
        trajectory_row.addWidget(self.trajectory, 1)
        trajectory_row.addWidget(browse)
        form.addRow("System name", self.name)
        form.addRow("AA structure", structure_row)
        form.addRow("AA trajectory", trajectory_row)
        input_layout.addWidget(form_box)

        stats_box = QtWidgets.QGroupBox("Trajectory summary")
        stats_layout = QtWidgets.QGridLayout(stats_box)
        self.stat_labels = {}
        for index, (key, label) in enumerate(
            [
                ("frames", "Stored frames"),
                ("steps", "Timestep span"),
                ("timesteps", "Timestep range"),
                ("stride", "Frame stride"),
                ("atoms", "Atoms per frame"),
                ("topology", "Topology"),
                ("box", "Initial box"),
                ("fields", "Available fields"),
                ("forces", "Force magnitudes"),
                ("size", "Trajectory size"),
            ]
        ):
            title = QtWidgets.QLabel(label)
            title.setObjectName("mutedLabel")
            value = QtWidgets.QLabel("—")
            value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self.stat_labels[key] = value
            row, column = divmod(index, 2)
            stats_layout.addWidget(title, row * 2, column)
            stats_layout.addWidget(value, row * 2 + 1, column)
        summary_layout.addWidget(stats_box)

        quality_box = QtWidgets.QGroupBox("Trajectory checks")
        quality_layout = QtWidgets.QVBoxLayout(quality_box)
        self.quality_table = QtWidgets.QTableWidget(0, 3)
        self.quality_table.setHorizontalHeaderLabels(["Check", "Status", "Details"])
        self.quality_table.verticalHeader().setVisible(False)
        self.quality_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.quality_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.quality_table.setMinimumHeight(0)
        self.quality_table.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        quality_header = self.quality_table.horizontalHeader()
        quality_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        quality_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        quality_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        quality_layout.addWidget(self.quality_table)
        checks_layout.addWidget(quality_box, 1)

        species_box = QtWidgets.QGroupBox("Detected molecular species")
        species_layout = QtWidgets.QVBoxLayout(species_box)
        self.detect_summary = QtWidgets.QLabel("Select the inputs, then run Stage 1 to detect species.")
        self.detect_summary.setWordWrap(True)
        self.species_table = QtWidgets.QTableWidget(0, 5)
        self.species_table.setHorizontalHeaderLabels(
            ["Species name", "Formula", "Count", "Canonical SMILES", "Bond orders"]
        )
        self.species_table.verticalHeader().setVisible(False)
        self.species_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.species_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.species_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.species_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.species_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.species_table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.species_table.horizontalHeader().setSectionResizeMode(
            4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.species_table.setMinimumHeight(0)
        self.species_table.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.species_table.itemSelectionChanged.connect(self._preview_selected_species)
        species_layout.addWidget(self.detect_summary)
        species_layout.addWidget(self.species_table)
        species_layout_tab.addWidget(species_box, 1)
        note = QtWidgets.QLabel(
            "Use a structure with known bonds when available. If it is missing, inspection can "
            "infer connectivity from the trajectory after your confirmation. "
            "Close keeps the selected inputs; Inspect inputs starts the analysis."
        )
        note.setWordWrap(True)
        note.setObjectName("note")
        input_layout.addWidget(note)
        input_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        input_buttons.rejected.connect(self.inputs_dialog.reject)
        inspect_button = input_buttons.addButton('Inspect inputs', QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        inspect_button.setObjectName('primaryButton')
        inspect_button.clicked.connect(self._inspect_from_inputs)
        input_layout.addWidget(input_buttons)
        details_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        details_buttons.rejected.connect(self.details_dialog.reject)
        details_layout.addWidget(details_buttons)
        summary_layout.addStretch()
        self.add_run_row("inspect", "Outputs: detected species, trajectory statistics, and explicit quality checks")
        # Setup and diagnostics live in dialogs; the main page is a viewing workspace.
        self.heading.setText('1 · Inspect')
        self.description.setText('Check inputs and detected species, then go to Map to define beads.')
        input_toolbar = QtWidgets.QHBoxLayout()
        self.inputs_button = QtWidgets.QPushButton('Load / change inputs…')
        self.inputs_button.clicked.connect(self.inputs_dialog.open)
        self.inspection_summary = QtWidgets.QLabel('Choose an AA structure and trajectory.')
        self.inspection_summary.setObjectName('pageDescription')
        self.inspection_summary.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.inspection_summary.setMinimumWidth(0)
        self.inspection_summary.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.check_summary = QtWidgets.QLabel('Not inspected')
        self.check_summary.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.details_button = QtWidgets.QPushButton('Details / checks…')
        self.details_button.clicked.connect(self._show_details)
        self._has_check_issues = False
        self.detail_tabs = tabs
        input_toolbar.addWidget(self.inputs_button)
        input_toolbar.addWidget(self.inspection_summary, 1)
        input_toolbar.addWidget(self.check_summary)
        input_toolbar.addWidget(self.details_button)
        self.page_layout.insertLayout(0, input_toolbar)
        self.trajectory_view = TrajectoryView(compact_controls=True)
        self.layout.addWidget(self.trajectory_view, 1)

    def _inspect_from_inputs(self):
        self.inputs_dialog.accept()
        self.runRequested.emit('inspect')

    def _show_details(self):
        if self._has_check_issues:
            self.detail_tabs.setCurrentIndex(1)
        self.details_dialog.show()
        self.details_dialog.raise_()

    def load_system(self, system: SystemSpec | None):
        enabled = system is not None
        self.setEnabled(enabled)
        if system is None:
            self.inputs_dialog.close()
            self.details_dialog.close()
            self.update_input_summary(None)
            return
        self.name.setText(system.name)
        self.structure_file.setText(system.structure_file)
        self.trajectory.setText(system.trajectory)
        self._load_species(system.detected_species)
        self._load_stats(system.trajectory_stats)
        self.update_input_summary(system)
        if system.structure_inferred and system.detected_species:
            self.detect_summary.setText(
                self.detect_summary.text()
                + " Connectivity was inferred from trajectory positions; inspect inferred_structure.data."
            )
        self.set_status(system.stages.get("inspect", "pending"))

    def update_input_summary(self, system: SystemSpec | None):
        self._has_check_issues = False
        self.check_summary.setStyleSheet('')
        if system is None:
            self.inspection_summary.setText('Choose a system to load its inputs.')
            self.inspection_summary.setToolTip('')
            self.check_summary.setText('Not inspected')
            return
        stats = system.trajectory_stats
        if system.detected_species and stats:
            summary = f"{system.name} · {stats.get('frames', '—')} frames · {len(system.detected_species)} species"
        else:
            summary = f'{system.name} · Inspect inputs to load the molecule views'
        self.inspection_summary.setText(summary)
        self.inspection_summary.setToolTip(
            f'{summary}\nAA structure: {system.structure_file or "Not selected"}\n'
            f'AA trajectory: {system.trajectory or "Not selected"}'
        )
        checks = stats.get('checks', [])
        failed = sum(c.get('status') == 'fail' for c in checks)
        warnings = sum(c.get('status') == 'warning' for c in checks)
        self._has_check_issues = bool(failed or warnings)
        if self._has_check_issues:
            issues = []
            if failed:
                issues.append(f'{failed} failed')
            if warnings:
                issues.append(f'{warnings} warnings')
            self.check_summary.setText('Checks: ' + ' · '.join(issues))
            colour = '#a23b3b' if failed else '#9a6a13'
            self.check_summary.setStyleSheet(f'color: {colour}; font-weight: 600;')
        elif system.stages.get('inspect') == 'failed':
            self.check_summary.setText('Inspection failed')
            self.check_summary.setStyleSheet('color: #a23b3b; font-weight: 600;')
        elif system.stages.get('inspect') != 'complete':
            self.check_summary.setText('Inspection needed')
        else:
            self.check_summary.setText('Checks passed' if checks else 'Checks unavailable')

    def apply(self, system: SystemSpec):
        system.name = self.name.text().strip() or "molecule"
        system.structure_file = self.structure_file.text().strip()
        system.trajectory = self.trajectory.text().strip()
        for row, species in enumerate(system.detected_species):
            item = self.species_table.item(row, 0)
            if item is not None:
                species["name"] = item.text().strip() or f"species_{row + 1}"

    def _load_species(self, species: list[dict]):
        self.species_table.blockSignals(True)
        self.species_table.setRowCount(len(species))
        for row, item in enumerate(species):
            values = [
                item.get("name", f"species_{row + 1}"),
                item.get("formula", ""),
                str(item.get("count", "")),
                item.get("smiles", ""),
                item.get("bond_order_status", ""),
            ]
            for column, value in enumerate(values):
                cell = QtWidgets.QTableWidgetItem(str(value))
                if column == 0:
                    cell.setData(QtCore.Qt.ItemDataRole.UserRole, item.get("id"))
                if column != 0:
                    cell.setFlags(cell.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.species_table.setItem(row, column, cell)
        self.species_table.blockSignals(False)
        total = sum(int(item.get("count", 0)) for item in species)
        if species:
            self.detect_summary.setText(
                f"{len(species)} species · {total} total molecules. Select a row to preview it."
            )
            self.species_table.selectRow(0)
        else:
            self.detect_summary.setText("Select the inputs, then run Stage 1 to detect species.")

    def _preview_selected_species(self):
        row = self.species_table.currentRow()
        if row < 0:
            return
        item = self.species_table.item(row, 0)
        if item is not None:
            self.speciesPreviewRequested.emit(str(item.data(QtCore.Qt.ItemDataRole.UserRole) or ""))

    def _load_stats(self, stats: dict):
        if not stats:
            for label in self.stat_labels.values():
                label.setText("—")
            self.quality_table.setRowCount(0)
            return
        box = stats.get("box_first", [])
        fields = []
        if stats.get("has_forces"):
            fields.append("forces")
        if stats.get("has_velocities"):
            fields.append("velocities")
        if stats.get("has_molecule_ids"):
            fields.append("molecule IDs")
        self.stat_labels["frames"].setText(str(stats.get("frames", "—")))
        self.stat_labels["steps"].setText(str(stats.get("timestep_span", "—")))
        self.stat_labels["timesteps"].setText(
            f"{stats.get('first_timestep', '—')} → {stats.get('last_timestep', '—')}"
        )
        self.stat_labels["stride"].setText(str(stats.get("frame_stride") or "variable / unknown"))
        self.stat_labels["atoms"].setText(str(stats.get("atoms", "—")))
        topology = stats.get("topology", {})
        self.stat_labels["topology"].setText(
            f"{topology.get('atoms', '—')} atoms · {topology.get('bonds', '—')} bonds"
            if topology else "—"
        )
        self.stat_labels["box"].setText(
            " × ".join(f"{float(value):.4g}" for value in box) if box else "—"
        )
        self.stat_labels["fields"].setText(", ".join(fields) if fields else "coordinates only")
        if stats.get("force_rms") is not None:
            self.stat_labels["forces"].setText(
                f"RMS {stats['force_rms']:.4g} · max {stats['force_max']:.4g}"
            )
        else:
            self.stat_labels["forces"].setText("—")
        size = int(stats.get("file_size_bytes", 0))
        self.stat_labels["size"].setText(_human_size(size) if size else "—")
        checks = stats.get("checks", [])
        self.quality_table.setRowCount(len(checks))
        status_labels = {"pass": "✓ Pass", "warning": "⚠ Warning", "fail": "✕ Fail"}
        status_colors = {"pass": "#2f7668", "warning": "#9a6a13", "fail": "#a23b3b"}
        for row, check in enumerate(checks):
            values = [
                check.get("name", ""),
                status_labels.get(check.get("status"), check.get("status", "")),
                check.get("detail", ""),
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if column == 1:
                    item.setForeground(QtGui.QColor(status_colors.get(check.get("status"), "#333333")))
                self.quality_table.setItem(row, column, item)


class MappingPage(PageBase):
    speciesPreviewRequested = QtCore.Signal(str)
    customRulePreviewRequested = QtCore.Signal()
    customRuleAccepted = QtCore.Signal()
    customRuleCancelled = QtCore.Signal()
    beadCreationModeChanged = QtCore.Signal(bool)
    descriptorSettingsChanged = QtCore.Signal()
    mappingModeChanged = QtCore.Signal()
    customRulesClearRequested = QtCore.Signal()
    customRuleCreationRequested = QtCore.Signal()
    customRuleDeleteRequested = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(
            "2 · Build CG model",
            "Define the atom-to-bead mapping, convert AA to CG, generate topology, and inspect histograms before force matching.",
            parent,
            scrollable=False,
        )
        self.enable_split_workspace()

        mode_box = QtWidgets.QGroupBox("Mapping mode")
        mode_layout = QtWidgets.QFormLayout(mode_box)
        self.mapping = QtWidgets.QComboBox()
        self.mapping.addItem("Manual custom beads", "manual_custom")
        self.mapping.addItem("United atom automatic", "united_atom")
        self.cores = QtWidgets.QSpinBox()
        self.cores.setRange(1, available_cores())
        self.cores.setValue(available_cores())
        self.cores_auto = QtWidgets.QCheckBox('Auto (all available)')
        self.cores_auto.setChecked(True)
        self.cores_auto.setToolTip(
            'Use all logical CPUs available to this process, including Linux affinity limits. '
            'Mapping, histograms and independent fits use up to this many workers, limited by '
            'the amount of work. Coupled fitting is one joint regression. '
            'LAMMPS parallelism is controlled separately by its launch command.')
        self.cores.setToolTip('Manual worker limit; disable Auto to change it.')
        self.cores.setEnabled(False)
        self.cores_auto.toggled.connect(self._cores_mode_changed)
        cores_row = QtWidgets.QWidget()
        cores_layout = QtWidgets.QHBoxLayout(cores_row)
        cores_layout.setContentsMargins(0, 0, 0, 0)
        cores_layout.addWidget(self.cores_auto)
        cores_layout.addWidget(self.cores)
        self.interaction_level = QtWidgets.QComboBox()
        self.interaction_level.addItem("Pair", "Pair")
        self.interaction_level.addItem("Bond", "Bond")
        self.interaction_level.addItem("Angle", "Angle")
        self.interaction_level.addItem("Dihedral", "Dihedral")
        mode_layout.addRow("Predefined set", self.mapping)
        mode_layout.addRow("Interaction level", self.interaction_level)
        mode_layout.addRow("Parallel jobs", cores_row)
        self.layout.addWidget(mode_box)

        custom_box = QtWidgets.QGroupBox("Custom bead rule builder")
        custom_box_layout = QtWidgets.QVBoxLayout(custom_box)
        bead_form = QtWidgets.QFormLayout()
        self.custom_bead_label = QtWidgets.QLineEdit("B1")
        self.custom_bead_label.setPlaceholderText("Bead label, e.g. A1 or RING")
        self.neighbor_depth = QtWidgets.QSpinBox()
        self.neighbor_depth.setRange(0, 3)
        self.neighbor_depth.setValue(1)
        bead_form.addRow("Bead label", self.custom_bead_label)
        bead_form.addRow("Descriptor depth", self.neighbor_depth)
        custom_box_layout.addLayout(bead_form)

        descriptor_grid = QtWidgets.QGridLayout()
        self.match_hydrogens = QtWidgets.QCheckBox("Hydrogen count")
        self.match_hydrogens.setChecked(True)
        self.match_aromatic = QtWidgets.QCheckBox("Aromatic")
        self.match_aromatic.setChecked(True)
        self.match_ring = QtWidgets.QCheckBox("Ring membership")
        self.match_ring.setChecked(True)
        self.match_hybridization = QtWidgets.QCheckBox("Hybridization")
        self.match_hybridization.setChecked(True)
        self.match_charge = QtWidgets.QCheckBox("Formal charge")
        self.match_charge.setChecked(True)
        self.match_neighbors = QtWidgets.QCheckBox("Neighbor shells")
        self.match_neighbors.setChecked(True)
        for index, checkbox in enumerate(
            [
                self.match_hydrogens,
                self.match_aromatic,
                self.match_ring,
                self.match_hybridization,
                self.match_charge,
                self.match_neighbors,
            ]
        ):
            descriptor_grid.addWidget(checkbox, index // 2, index % 2)
        custom_box_layout.addLayout(descriptor_grid)

        action_row = QtWidgets.QHBoxLayout()
        self.pick_rule_button = QtWidgets.QPushButton("Create new bead")
        self.pick_rule_button.setCheckable(True)
        self.pick_rule_button.setObjectName("primaryButton")
        self.save_rule_button = QtWidgets.QPushButton("Finish bead")
        self.cancel_rule_button = QtWidgets.QPushButton("Cancel")
        action_row.addWidget(self.pick_rule_button)
        action_row.addWidget(self.save_rule_button)
        action_row.addWidget(self.cancel_rule_button)
        custom_box_layout.addLayout(action_row)

        self.custom_status = QtWidgets.QLabel("Inspection mode. Click Create new bead to select atoms for a bead type.")
        self.custom_status.setObjectName("note")
        self.custom_status.setWordWrap(True)
        custom_box_layout.addWidget(self.custom_status)

        self.descriptor_table = QtWidgets.QTableWidget(0, 2)
        self.descriptor_table.setHorizontalHeaderLabels(["Descriptor", "Value"])
        self.descriptor_table.verticalHeader().setVisible(False)
        self.descriptor_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        descriptor_header = self.descriptor_table.horizontalHeader()
        descriptor_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        descriptor_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.descriptor_table.setMinimumHeight(118)
        custom_box_layout.addWidget(self.descriptor_table)

        self.matches_table = QtWidgets.QTableWidget(0, 3)
        self.matches_table.setHorizontalHeaderLabels(["Match", "Center", "Atoms"])
        self.matches_table.verticalHeader().setVisible(False)
        self.matches_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        matches_header = self.matches_table.horizontalHeader()
        matches_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        matches_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        matches_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.matches_table.setMinimumHeight(120)
        custom_box_layout.addWidget(self.matches_table)
        self.layout.addWidget(custom_box)
        custom_box.hide()

        editor_box = QtWidgets.QGroupBox("Custom bead mapping")
        editor_layout = QtWidgets.QVBoxLayout(editor_box)
        editor_note = QtWidgets.QLabel(
            "Create bead rules in a focused editor where the atom selection, chemical "
            "description, equivalent matches, and conflicts stay visible together."
        )
        editor_note.setObjectName("note")
        editor_note.setWordWrap(True)
        self.open_bead_editor_button = QtWidgets.QPushButton("Create bead rule…")
        self.open_bead_editor_button.setObjectName("primaryButton")
        editor_layout.addWidget(editor_note)
        editor_layout.addWidget(
            self.open_bead_editor_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.layout.addWidget(editor_box)

        saved_box = QtWidgets.QGroupBox("Saved custom bead rules")
        saved_layout = QtWidgets.QVBoxLayout(saved_box)
        saved_actions = QtWidgets.QHBoxLayout()
        saved_actions.addStretch()
        self.delete_rule_button = QtWidgets.QPushButton("Delete selected")
        self.delete_rule_button.setEnabled(False)
        self.clear_rules_button = QtWidgets.QPushButton("Clear species rules")
        saved_actions.addWidget(self.delete_rule_button)
        saved_actions.addWidget(self.clear_rules_button)
        saved_layout.addLayout(saved_actions)
        self.saved_rules_table = QtWidgets.QTableWidget(0, 5)
        self.saved_rules_table.setHorizontalHeaderLabels(["Bead", "Species", "Matches", "Atoms", "Signature"])
        self.saved_rules_table.verticalHeader().setVisible(False)
        self.saved_rules_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.saved_rules_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.saved_rules_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        saved_header = self.saved_rules_table.horizontalHeader()
        for column in range(4):
            saved_header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        saved_header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.saved_rules_table.setMinimumHeight(0)
        self.saved_rules_table.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        saved_layout.addWidget(self.saved_rules_table)
        self.layout.addWidget(saved_box, 1)

        self.add_run_row(
            "cgmodel",
            "Outputs: prepared mapping files, CG trajectory/topology, histogram CSV/PNG files, and fit_plan.json",
        )

        self.pick_rule_button.toggled.connect(self._bead_creation_toggled)
        self.save_rule_button.clicked.connect(self.customRuleAccepted.emit)
        self.cancel_rule_button.clicked.connect(self._cancel_bead_creation)
        self.clear_rules_button.clicked.connect(self.customRulesClearRequested.emit)
        self.delete_rule_button.clicked.connect(self._delete_selected_rule)
        self.saved_rules_table.itemSelectionChanged.connect(
            self._saved_rule_selection_changed
        )
        self.open_bead_editor_button.clicked.connect(self.customRuleCreationRequested.emit)
        self.mapping.currentIndexChanged.connect(self._mapping_mode_changed)
        self.neighbor_depth.valueChanged.connect(self.descriptorSettingsChanged.emit)
        self.custom_bead_label.textChanged.connect(self.descriptorSettingsChanged.emit)
        for checkbox in [
            self.match_hydrogens,
            self.match_aromatic,
            self.match_ring,
            self.match_hybridization,
            self.match_charge,
            self.match_neighbors,
        ]:
            checkbox.toggled.connect(self.descriptorSettingsChanged.emit)

        self._mapping_mode_changed()

        self.settings_dialog = QtWidgets.QDialog(self)
        self.settings_dialog.setWindowTitle('Mapping settings and saved bead rules')
        self.settings_dialog.resize(850, 720)
        settings_layout = QtWidgets.QVBoxLayout(self.settings_dialog)
        settings_layout.addWidget(self.workspace_splitter.widget(0))
        # The shared species-canvas toolbar hosts Create bead rule and its
        # View menu exposes this settings dialog (wired by MainWindow).
        self.trajectory_view = TrajectoryView(mapping=True)
        self.trajectory_view.scope.setCurrentIndex(2)
        self.workspace_splitter.addWidget(self.trajectory_view)
        self.workspace_splitter.setStretchFactor(0, 1)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([600, 600])
        self.heading.setText('2 · Map')
        self.description.setText('Define beads and build the mapped trajectory, then review bounds in Configure fit.')

    def _cores_mode_changed(self, automatic):
        self.cores.setEnabled(not automatic)
        if automatic:
            self.cores.setValue(available_cores())

    def load_system(self, system: SystemSpec | None):
        self.setEnabled(system is not None)
        if system is None:
            return
        self.cores.setMaximum(max(available_cores(), system.cores))
        self.cores_auto.setChecked(system.cores_auto)
        self.cores.setValue(system.effective_cores if system.cores_auto else system.cores)
        self.cores.setEnabled(not system.cores_auto)
        index = self.interaction_level.findData(system.interaction_level)
        self.interaction_level.setCurrentIndex(index if index >= 0 else 2)
        mode_index = self.mapping.findData(getattr(system, "mapping_mode", "manual_custom"))
        self.mapping.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        self._load_custom_rules(system)
        self._mapping_mode_changed()

    def apply(self, system: SystemSpec):
        system.cores = self.cores.value()
        system.cores_auto = self.cores_auto.isChecked()
        system.interaction_level = str(self.interaction_level.currentData())
        system.mapping_mode = self.mapping_mode()

    def mapping_mode(self) -> str:
        return str(self.mapping.currentData() or "manual_custom")

    def descriptor_settings(self) -> dict:
        return {
            "neighbor_depth": self.neighbor_depth.value(),
            "match_hydrogens": self.match_hydrogens.isChecked(),
            "match_aromatic": self.match_aromatic.isChecked(),
            "match_ring": self.match_ring.isChecked(),
            "match_hybridization": self.match_hybridization.isChecked(),
            "match_charge": self.match_charge.isChecked(),
            "match_neighbors": self.match_neighbors.isChecked(),
        }

    def bead_label(self) -> str:
        return self.custom_bead_label.text().strip()

    def bead_creation_active(self) -> bool:
        return self.pick_rule_button.isChecked()

    def set_bead_creation_active(self, active: bool):
        if self.pick_rule_button.isChecked() != bool(active):
            self.pick_rule_button.setChecked(bool(active))
        else:
            self._update_bead_creation_controls()

    def set_custom_rule_preview(self, preview: dict | None):
        self.descriptor_table.setRowCount(0)
        self.matches_table.setRowCount(0)
        if not preview:
            if self.bead_creation_active():
                self.custom_status.setText("Bead creation mode. Select a connected atom group on the canvas, then click Finish bead.")
            else:
                self.custom_status.setText("Inspection mode. Click Create new bead to select atoms for a bead type.")
            return
        if preview.get("error"):
            self.custom_status.setText(str(preview["error"]))
            return
        descriptor = preview.get("descriptor", {})
        self.descriptor_table.setRowCount(len(descriptor))
        for row, key in enumerate(sorted(descriptor)):
            self.descriptor_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(key)))
            self.descriptor_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(descriptor[key])))
        matches = preview.get("matches", [])
        self.matches_table.setRowCount(len(matches))
        for row, match in enumerate(matches):
            self.matches_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))
            self.matches_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(match.get("center", ""))))
            atoms = ", ".join(str(atom) for atom in match.get("atoms", []))
            self.matches_table.setItem(row, 2, QtWidgets.QTableWidgetItem(atoms))
        self.custom_status.setText(
            f"{preview.get('bead_label', 'bead')} matches {len(matches)} equivalent instance(s) in the selected species."
        )

    def _load_custom_rules(self, system: SystemSpec):
        rules = list(system.custom_mapping_rules)
        self.saved_rules_table.setRowCount(len(rules))
        for row, rule in enumerate(rules):
            atom_count = len(rule.get("prototype_atoms", []))
            values = [
                rule.get("bead_label", ""),
                rule.get("species_name", rule.get("species_id", "")),
                len(rule.get("matches", [])),
                atom_count,
                rule.get("signature", ""),
            ]
            for column, value in enumerate(values):
                self.saved_rules_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.delete_rule_button.setEnabled(False)

    def _saved_rule_selection_changed(self):
        self.delete_rule_button.setEnabled(
            self.mapping_mode() == "manual_custom"
            and self.saved_rules_table.currentRow() >= 0
        )

    def _delete_selected_rule(self):
        row = self.saved_rules_table.currentRow()
        if row >= 0:
            self.customRuleDeleteRequested.emit(row)

    def _mapping_mode_changed(self):
        manual = self.mapping_mode() == "manual_custom"
        for widget in [
            self.pick_rule_button,
            self.open_bead_editor_button,
            self.delete_rule_button,
            self.saved_rules_table,
            self.clear_rules_button,
        ]:
            widget.setEnabled(manual)
        self.delete_rule_button.setEnabled(
            manual and self.saved_rules_table.currentRow() >= 0
        )
        if not manual and self.pick_rule_button.isChecked():
            self.pick_rule_button.setChecked(False)
        self._update_bead_creation_controls()
        self.mappingModeChanged.emit()

    def _bead_creation_toggled(self, active: bool):
        self._update_bead_creation_controls()
        self.beadCreationModeChanged.emit(bool(active))

    def _cancel_bead_creation(self):
        self.pick_rule_button.setChecked(False)
        self.customRuleCancelled.emit()

    def _update_bead_creation_controls(self):
        manual = self.mapping_mode() == "manual_custom"
        active = manual and self.pick_rule_button.isChecked()
        self.pick_rule_button.setText("Creating bead..." if active else "Create new bead")
        for widget in [
            self.custom_bead_label,
            self.neighbor_depth,
            self.match_hydrogens,
            self.match_aromatic,
            self.match_ring,
            self.match_hybridization,
            self.match_charge,
            self.match_neighbors,
            self.save_rule_button,
            self.cancel_rule_button,
            self.descriptor_table,
            self.matches_table,
        ]:
            widget.setEnabled(active)
        if not manual:
            self.custom_status.setText("United atom automatic mapping is active.")
        elif active and self.descriptor_table.rowCount() == 0 and self.matches_table.rowCount() == 0:
            self.custom_status.setText("Bead creation mode. Select a connected atom group on the canvas, then click Finish bead.")
        elif not active and self.descriptor_table.rowCount() == 0 and self.matches_table.rowCount() == 0:
            self.custom_status.setText("Inspection mode. Click Create new bead to select atoms for a bead type.")

class BeadRuleDialog(QtWidgets.QDialog):
    """Focused atom-selection and reusable bead-rule editor."""

    previewRequested = QtCore.Signal()
    createRequested = QtCore.Signal()

    def __init__(self, species: dict, suggested_label: str, parent=None):
        super().__init__(parent)
        self.species = species
        self.setWindowTitle(f"Create bead rule · {species.get('name', 'species')}")
        self.setModal(True)
        self.resize(1260, 780)
        self.setMinimumSize(980, 650)

        root = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel("Create bead rule")
        heading.setObjectName("pageTitle")
        context = QtWidgets.QLabel(
            f"{species.get('name', 'species')}  ·  {species.get('formula', '')}  ·  "
            f"{species.get('smiles', '')}"
        )
        context.setObjectName("pageDescription")
        root.addWidget(heading)
        root.addWidget(context)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        canvas_box = QtWidgets.QGroupBox("1 · Select one connected atom group")
        canvas_layout = QtWidgets.QVBoxLayout(canvas_box)
        instruction = QtWidgets.QLabel(
            "Click atoms to select the prototype bead. Existing mapped atoms are locked; "
            "the proposed rule is applied to every equivalent match."
        )
        instruction.setObjectName("note")
        instruction.setWordWrap(True)
        canvas_layout.addWidget(instruction)
        self.canvas = MoleculeCanvas()
        self.canvas.setMinimumSize(540, 470)
        self.canvas.set_selection_only(True)
        self.canvas.set_united_atom_mode(False)
        self.canvas.set_atom_circle_mode(True)
        self.canvas.show_h = True
        self.canvas.show_carbon_dots = True
        self.canvas.load_smiles(species.get("smiles", ""), explicit_hydrogens=True)
        canvas_layout.addWidget(self.canvas, 1)
        selection_row = QtWidgets.QHBoxLayout()
        self.selection_summary = QtWidgets.QLabel("No atoms selected")
        self.selection_summary.setObjectName("sectionTitle")
        self.clear_button = QtWidgets.QPushButton("Clear selection")
        self.clear_button.clicked.connect(self._clear_selection)
        selection_row.addWidget(self.selection_summary, 1)
        selection_row.addWidget(self.clear_button)
        canvas_layout.addLayout(selection_row)

        details = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(details)
        details_layout.setContentsMargins(10, 0, 0, 0)

        identity_box = QtWidgets.QGroupBox("2 · Describe the bead")
        identity_layout = QtWidgets.QFormLayout(identity_box)
        self.custom_bead_label = QtWidgets.QLineEdit(suggested_label)
        self.custom_bead_label.setPlaceholderText("Bead label, e.g. A1 or RING")
        identity_layout.addRow("Bead label", self.custom_bead_label)
        details_layout.addWidget(identity_box)

        self.advanced_toggle = QtWidgets.QToolButton()
        self.advanced_toggle.setText("Advanced matching options")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.advanced_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        details_layout.addWidget(self.advanced_toggle)

        self.advanced_panel = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(12, 0, 0, 6)
        depth_form = QtWidgets.QFormLayout()
        self.neighbor_depth = QtWidgets.QSpinBox()
        self.neighbor_depth.setRange(0, 3)
        self.neighbor_depth.setValue(1)
        depth_form.addRow("Descriptor depth", self.neighbor_depth)
        advanced_layout.addLayout(depth_form)
        option_grid = QtWidgets.QGridLayout()
        self.match_hydrogens = QtWidgets.QCheckBox("Hydrogen count")
        self.match_aromatic = QtWidgets.QCheckBox("Aromatic")
        self.match_ring = QtWidgets.QCheckBox("Ring membership")
        self.match_hybridization = QtWidgets.QCheckBox("Hybridization")
        self.match_charge = QtWidgets.QCheckBox("Formal charge")
        self.match_neighbors = QtWidgets.QCheckBox("Neighbor shells")
        self._match_options = [
            self.match_hydrogens,
            self.match_aromatic,
            self.match_ring,
            self.match_hybridization,
            self.match_charge,
            self.match_neighbors,
        ]
        for index, checkbox in enumerate(self._match_options):
            checkbox.setChecked(True)
            option_grid.addWidget(checkbox, index // 2, index % 2)
        advanced_layout.addLayout(option_grid)
        self.advanced_panel.hide()
        details_layout.addWidget(self.advanced_panel)

        preview_box = QtWidgets.QGroupBox("3 · Review the proposed rule")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        self.custom_status = QtWidgets.QLabel("Select atoms to preview the bead rule.")
        self.custom_status.setObjectName("note")
        self.custom_status.setWordWrap(True)
        preview_layout.addWidget(self.custom_status)
        self.descriptor_table = QtWidgets.QTableWidget(0, 2)
        self.descriptor_table.setHorizontalHeaderLabels(["Description", "Value"])
        self.descriptor_table.verticalHeader().setVisible(False)
        self.descriptor_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.descriptor_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.descriptor_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.descriptor_table.setMinimumHeight(145)
        preview_layout.addWidget(self.descriptor_table)
        self.matches_table = QtWidgets.QTableWidget(0, 3)
        self.matches_table.setHorizontalHeaderLabels(
            ["Match", "Center atom", "Atom indices"]
        )
        self.matches_table.verticalHeader().setVisible(False)
        self.matches_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.matches_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.matches_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.matches_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.matches_table.setMinimumHeight(145)
        preview_layout.addWidget(self.matches_table)
        details_layout.addWidget(preview_box, 1)

        splitter.addWidget(canvas_box)
        splitter.addWidget(details)
        splitter.setSizes([700, 500])
        root.addWidget(splitter, 1)

        buttons = QtWidgets.QDialogButtonBox()
        self.create_button = buttons.addButton(
            "Create bead rule", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.create_button.setObjectName("primaryButton")
        self.create_button.setEnabled(False)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        self.create_button.clicked.connect(self.createRequested.emit)
        root.addWidget(buttons)

        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.custom_bead_label.textChanged.connect(self.previewRequested.emit)
        self.neighbor_depth.valueChanged.connect(self.previewRequested.emit)
        for checkbox in self._match_options:
            checkbox.toggled.connect(self.previewRequested.emit)

    def _toggle_advanced(self, visible: bool):
        self.advanced_panel.setVisible(visible)
        self.advanced_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if visible
            else QtCore.Qt.ArrowType.RightArrow
        )

    def _clear_selection(self):
        self.canvas.clear_selection()
        self.previewRequested.emit()

    def bead_label(self) -> str:
        return self.custom_bead_label.text().strip()

    def descriptor_settings(self) -> dict:
        return {
            "neighbor_depth": self.neighbor_depth.value(),
            "match_hydrogens": self.match_hydrogens.isChecked(),
            "match_aromatic": self.match_aromatic.isChecked(),
            "match_ring": self.match_ring.isChecked(),
            "match_hybridization": self.match_hybridization.isChecked(),
            "match_charge": self.match_charge.isChecked(),
            "match_neighbors": self.match_neighbors.isChecked(),
        }

    def set_preview(self, preview: dict | None):
        self.descriptor_table.setRowCount(0)
        self.matches_table.setRowCount(0)
        self.create_button.setEnabled(False)
        selected_count = len(self.canvas.selected_atoms)
        self.selection_summary.setText(
            "No atoms selected"
            if selected_count == 0
            else f"{selected_count} atom(s) selected"
        )
        if not preview:
            self.custom_status.setText("Select atoms to preview the bead rule.")
            self.create_button.setText("Create bead rule")
            return
        if preview.get("error"):
            self.custom_status.setText(str(preview["error"]))
            self.create_button.setText("Create bead rule")
            return

        descriptor = preview.get("descriptor", {})
        visible_keys = ["formula", "atom_count", "heavy_count", "elements", "bonds", "smarts"]
        for key in visible_keys:
            if key not in descriptor:
                continue
            row = self.descriptor_table.rowCount()
            self.descriptor_table.insertRow(row)
            self.descriptor_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(key.replace("_", " ").title())
            )
            self.descriptor_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(str(descriptor[key]))
            )
        matches = preview.get("matches", [])
        self.matches_table.setRowCount(len(matches))
        for row, match in enumerate(matches):
            self.matches_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))
            self.matches_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(str(match.get("center", "")))
            )
            atoms = ", ".join(str(atom) for atom in match.get("atoms", []))
            self.matches_table.setItem(row, 2, QtWidgets.QTableWidgetItem(atoms))
        label = preview.get("bead_label", "bead")
        self.custom_status.setText(
            f"{label} will create {len(matches)} equivalent bead instance(s) in this species."
        )
        self.create_button.setText(
            f"Create bead rule — {len(matches)} match(es)"
        )
        self.create_button.setEnabled(bool(matches) and bool(self.bead_label()))


class SimpleStagePage(PageBase):
    def __init__(self, stage: str, title: str, description: str, output: str, parent=None):
        super().__init__(title, description, parent)
        self.add_run_row(stage, output)


class ForceMatchPage(PageBase):
    """Force-matching action plus a read-only reminder of Stage 2 inputs."""

    def __init__(self, parent=None):
        super().__init__(
            "4 · Fit force field",
            "Run force matching with the Configure fit settings. Review raw results, then go to Prepare tables.",
            parent,
            scrollable=False,
        )
        method_layout = QtWidgets.QHBoxLayout()
        self.method = QtWidgets.QComboBox()
        self.method.addItem("Independent", "independent")
        self.method.addItem("Coupled", "coupled")
        self.method.setToolTip(
            "Coupled fits every selected interaction in one OpenMSCG regression. "
            "Independent reproduces the previous one-interaction-at-a-time workflow."
        )
        method_layout.addWidget(QtWidgets.QLabel('Method'))
        method_layout.addWidget(self.method)
        method_layout.addStretch()
        details_button = QtWidgets.QToolButton(text='Mapping / inputs…')
        method_layout.addWidget(details_button)
        self.layout.addLayout(method_layout)
        self.input_details = QtWidgets.QDialog(self)
        self.input_details.setWindowTitle('Force matching — mapping and required inputs')
        details_layout = QtWidgets.QVBoxLayout(self.input_details)
        details_button.clicked.connect(self.input_details.show)

        inherited_box = QtWidgets.QGroupBox("Inherited from Stage 2 Mapping")
        inherited = QtWidgets.QFormLayout(inherited_box)
        self.inherited_labels = {}
        for key, title in [
            ("mapping", "Mapping mode"),
            ("interaction", "Interaction level"),
            ("cores", "Configured cores"),
            ("species", "Molecular species"),
            ("beads", "Bead definitions"),
            ("coverage", "Mapping coverage"),
        ]:
            value = QtWidgets.QLabel("—")
            value.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            value.setWordWrap(True)
            self.inherited_labels[key] = value
            inherited.addRow(title, value)
        reminder = QtWidgets.QLabel(
            "These values are read-only here. Return to Stage 2 Mapping to change them; "
            "rerunning Stage 2 invalidates this force-matching stage."
        )
        reminder.setObjectName("note")
        reminder.setWordWrap(True)
        inherited.addRow(reminder)
        inherited_box.setEnabled(False)
        details_layout.addWidget(inherited_box)

        output_box = QtWidgets.QGroupBox("Required Stage 2 outputs")
        output_layout = QtWidgets.QFormLayout(output_box)
        self.output_labels = {}
        for key, title in [
            ("topology", "CG topology"),
            ("trajectory", "Mapped CG trajectory"),
            ("fit_plan", "Histogram fit plan"),
        ]:
            value = QtWidgets.QLabel("Not generated")
            self.output_labels[key] = value
            output_layout.addRow(title, value)
        details_layout.addWidget(output_box)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.input_details.reject)
        details_layout.addWidget(close)
        self.review = FitReview()
        self.layout.addWidget(self.review, 1)
        self.method.currentIndexChanged.connect(lambda *_: self.review.set_method(self.method.currentData()))
        self.add_run_row(
            "force",
            "Outputs: raw .p models, raw .table files, and table plots",
        )

    def load_system(self, system: SystemSpec | None, system_root: str | Path | None = None):
        self.setEnabled(system is not None)
        if system is None:
            self.method.setCurrentIndex(0)
            for label in self.inherited_labels.values():
                label.setText("—")
            for label in self.output_labels.values():
                label.setText("Not generated")
            self.review.load(None)
            self.input_details.close()
            return

        method_index = self.method.findData(system.force_match_method)
        self.method.setCurrentIndex(max(0, method_index))

        manual = system.mapping_mode == "manual_custom"
        self.inherited_labels["mapping"].setText(
            "Manual custom beads" if manual else "United atom automatic"
        )
        self.inherited_labels["interaction"].setText(system.interaction_level)
        self.inherited_labels["cores"].setText(
            f"{system.effective_cores} ({'Auto' if system.cores_auto else 'Manual'}) · "
            + ("Map histograms; coupled fit is one joint regression"
               if system.force_match_method == "coupled" else "independent fitting jobs")
        )
        self.inherited_labels["species"].setText(
            f"{len(system.detected_species)} detected species"
            if system.detected_species
            else "Stage 1 inspection incomplete"
        )

        if manual:
            rules = list(system.custom_mapping_rules)
            instances = sum(len(rule.get("matches", [])) for rule in rules)
            self.inherited_labels["beads"].setText(
                f"{len(rules)} saved rule(s) · {instances} bead instance(s)"
            )
            mapped = 0
            total = 0
            for species in system.detected_species:
                species_id = species.get("id", "")
                species_total = int(species.get("atom_count", 0))
                covered = {
                    int(atom_id)
                    for rule in rules
                    if rule.get("species_id") == species_id
                    for match in rule.get("matches", [])
                    for atom_id in match.get("atoms", [])
                    if 0 <= int(atom_id) < species_total
                }
                mapped += len(covered)
                total += species_total
            self.inherited_labels["coverage"].setText(
                f"{mapped}/{total} representative atoms mapped"
                if total
                else "No inspected atoms"
            )
        else:
            ua_sites = sum(
                int(species.get("heavy_atom_count", 0))
                for species in system.detected_species
            )
            self.inherited_labels["beads"].setText(
                f"{ua_sites} automatic UA site(s) across species templates"
            )
            self.inherited_labels["coverage"].setText("Automatic united-atom mapping")

        root = Path(system_root) if system_root else None
        paths = {
            "topology": root / "output" / "work" / "top.in" if root else None,
            "trajectory": root / "output" / "work" / "types_cg.lammpstrj" if root else None,
            "fit_plan": root / "output" / "work" / "fit_plan.json" if root else None,
        }
        for key, path in paths.items():
            self.output_labels[key].setText(
                f"✓ Ready · {path.name}" if path is not None and path.is_file() else "Not generated"
            )
        self.review.load(root, system.fit_overrides, system.stages.get('cgmodel') == 'complete',
                         system.stages.get('force', 'pending'), self.method.currentData())

    def set_status(self, status, running=False):
        super().set_status(status, running)
        if hasattr(self, 'review'):
            self.review.set_status(status)

    def apply(self, system: SystemSpec):
        method = self.method.currentData()
        system.force_match_method = (
            str(method) if method in {"coupled", "independent"} else "independent"
        )


class TablesPage(PageBase):
    def __init__(self, parent=None):
        super().__init__(
            "5 · Prepare tables",
            "Compare processed curves with raw fits, then run Prepare tables. Next: generate the LAMMPS setup.",
            parent,
            scrollable=False,
        )
        self.explorer = TableProcessingExplorer()
        self.layout.addWidget(self.explorer, 1)
        self.add_run_row(
            "postprocess",
            "Outputs: output/processed_tables/1_all, 2_accepted, and rejection diagnostics",
        )

    def load_system_root(self, root, overrides=None, current=False):
        self.explorer.load_system_root(root, overrides, current)

    def set_status(self, value, running=False):
        super().set_status(value, running)
        self.explorer.outputs_current = value == 'complete'

    def reload_results(self):
        self.explorer.reload()

    def show_accepted(self):
        self.explorer.show_accepted()


class LammpsPage(PageBase):
    smokeTestRequested = QtCore.Signal()
    commandChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(
            "6 · LAMMPS",
            "Generate and smoke-test the setup (no dynamics). Next: optional Refine, or Validate an existing CG simulation.",
            parent,
            scrollable=False,
        )
        self.system_root: Path | None = None
        self._files_ready = False
        self._runner_busy = False

        smoke_box = QtWidgets.QGroupBox("Zero-step LAMMPS smoke test")
        smoke_layout = QtWidgets.QGridLayout(smoke_box)
        self.smoke_status = QtWidgets.QLabel("Pending")
        self.smoke_status.setObjectName("stageStatus")
        self.smoke_explanation = QtWidgets.QLabel(
            "Copies output/lammps to smoke_test, replaces the production section with run 0, and launches LAMMPS there."
        )
        self.smoke_explanation.setWordWrap(True)
        self.smoke_explanation.setObjectName("note")
        self.command = QtWidgets.QLineEdit()
        self.command.setPlaceholderText("mpiexec -np 16 lmp -in input.data")
        self.command.setToolTip("Shared with the command in Refine / Property optimization")
        self.command.editingFinished.connect(
            lambda: self.commandChanged.emit(self.command.text().strip())
        )
        self.smoke_button = QtWidgets.QPushButton("Run LAMMPS smoke test")
        self.smoke_button.clicked.connect(self.smokeTestRequested.emit)
        smoke_layout.addWidget(self.smoke_status, 0, 0)
        smoke_layout.addWidget(self.smoke_explanation, 0, 1, 1, 2)
        smoke_layout.addWidget(QtWidgets.QLabel("Command"), 1, 0)
        smoke_layout.addWidget(self.command, 1, 1)
        smoke_layout.addWidget(self.smoke_button, 1, 2)
        self.layout.addWidget(smoke_box)

        files = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        files.setChildrenCollapsible(False)
        self.input_preview = self._file_preview(files, "input.data")
        self.structure_preview = self._file_preview(files, "structure.data")
        files.setSizes([600, 600])
        self.layout.addWidget(files, 1)

        self.add_run_row(
            "lammps",
            "Outputs: output/lammps containing structure.data, input.data, accepted tables, and bead signatures",
        )
        self.run_button.setText("Generate LAMMPS setup")
        self._update_smoke_button()

    @staticmethod
    def _file_preview(parent, title: str) -> QtWidgets.QPlainTextEdit:
        box = QtWidgets.QGroupBox(title)
        parent.addWidget(box)
        layout = QtWidgets.QVBoxLayout(box)
        preview = QtWidgets.QPlainTextEdit()
        preview.setReadOnly(True)
        preview.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        preview.setPlaceholderText(f"Generate the LAMMPS setup to create {title}.")
        preview.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        layout.addWidget(preview)
        return preview

    def load_system_root(self, root, command: list[str] | None = None):
        self.system_root = Path(root) if root else None
        self.command.setText(display_command(command or []))
        self.reload_files()

    def reload_files(self):
        folder = self.system_root / "output" / "lammps" if self.system_root else None
        input_path = folder / "input.data" if folder else None
        structure_path = folder / "structure.data" if folder else None
        self._load_preview(self.input_preview, input_path)
        self._load_preview(self.structure_preview, structure_path)
        self._files_ready = bool(
            input_path is not None
            and structure_path is not None
            and input_path.is_file()
            and structure_path.is_file()
        )
        self._update_smoke_button()

    @staticmethod
    def _load_preview(preview: QtWidgets.QPlainTextEdit, path: Path | None):
        if path is None or not path.is_file():
            preview.clear()
            return
        try:
            preview.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            preview.setPlainText(f"Could not read {path}:\n{exc}")

    def set_status(self, value: str, running: bool = False):
        super().set_status(value, running)
        self.run_button.setText("Generating…" if running else "Generate LAMMPS setup")

    def set_smoke_status(self, value: str, running: bool = False):
        labels = {
            "pending": "Not tested",
            "running": "Testing…",
            "complete": "Passed",
            "failed": "Failed",
        }
        self._runner_busy = running
        self.smoke_status.setText(labels.get(value, value.title()))
        self.smoke_status.setProperty("state", value)
        self.smoke_status.style().unpolish(self.smoke_status)
        self.smoke_status.style().polish(self.smoke_status)
        self.smoke_button.setText(
            "Running smoke test…" if value == "running" else "Run LAMMPS smoke test"
        )
        self._update_smoke_button()

    def _update_smoke_button(self):
        self.smoke_button.setEnabled(
            self._files_ready and bool(self.command.text().strip()) and not self._runner_busy
        )


class ValidationPage(PageBase):
    aaPropertiesRequested = QtCore.Signal()
    samplingChanged = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(
            "8 · Validate",
            "Compare AA and CG properties and distributions. Load AA properties if missing; revisit fitting or refinement if needed.",
            parent,
            scrollable=False,
        )
        self.viewer = ValidationViewer()
        self.viewer.aaPropertiesRequested.connect(self.aaPropertiesRequested.emit)
        self.viewer.sampling_button.clicked.connect(self._edit_sampling)
        self._sampling = normalize_sampling()
        self._has_system = False
        self.viewer.sampling_button.setEnabled(False)
        self.viewer.setMinimumHeight(0)
        self.result_tabs = self.viewer.tabs
        self.layout.addWidget(self.viewer, 1)
        self.comparison = TrajectoryComparison()
        self.trajectory_view = self.comparison.aa_view
        self.cg_view = self.comparison.cg_view
        self.result_tabs.addTab(self.comparison, '3D visuals')
        self.add_run_row(
            "validate",
            "Outputs: validation/metrics.json; compares the selected system's mapped-AA and best available CG trajectories",
        )

    def load_system(self, system: SystemSpec | None, root=None):
        self._has_system = system is not None
        self._sampling = normalize_sampling(system.validation_sampling if system is not None else None)
        self.viewer.sampling_button.setEnabled(self._has_system)
        self.viewer.sampling_button.setToolTip(
            f"Up to {self._sampling['maximum_frames']} evenly spaced frames per trajectory; "
            f"discard initial {self._sampling['discard_percent']:g}%. Changes apply on the next validation run.")
        self.viewer.load_system(
            root,
            system.aa_properties_file if system is not None else None,
            self._sampling,
        )

    def _edit_sampling(self):
        if not self._has_system:
            return
        dialog = ValidationSamplingDialog(self._sampling, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            settings = dialog.settings()
            if settings != self._sampling:
                self.samplingChanged.emit(settings)

    def load_system_root(self, root):
        self.viewer.load_system_root(root)

    def reload_results(self):
        self.viewer.reload()


class PropertySettingsDialog(QtWidgets.QDialog):
    """Detailed editor for one property-scoring definition."""

    OPTIONAL_FIELDS = {"target", "min", "max", "max_abs_drift"}

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.settings = dict(settings)
        definition = property_definition(settings["column"])
        self.setWindowTitle(f"Configure {definition['label']}")
        self.setMinimumWidth(520)
        layout = QtWidgets.QVBoxLayout(self)

        heading = QtWidgets.QLabel(f"{definition['label']}  ·  {settings['column']}")
        heading.setObjectName("sectionTitle")
        explanation = QtWidgets.QLabel(
            "Each statistical term is divided by its tolerance before weighting. "
            "Hard limits add the configured penalty when violated. Blank optional fields are disabled."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("note")
        layout.addWidget(heading)
        layout.addWidget(explanation)

        form = QtWidgets.QFormLayout()
        self.fields = {}
        definitions = [
            ("target", "Target / reference", "Blank uses the first retained production sample, after discarding the initial fraction"),
            ("weight", "Objective weight", "Multiplier applied after normalized scoring"),
            ("tail_scale", "Tail-mean tolerance", "Scale for tail mean minus reference"),
            ("drift_scale", "Final-drift tolerance", "Scale for final value minus reference"),
            ("std_scale", "Noise tolerance", "Scale for standard deviation"),
            ("slope_scale", "Slope tolerance", "Scale for cumulative fitted slope"),
            ("skip_fraction", "Initial production fraction skipped", "Between 0 and less than 1; excludes production samples from scoring, not extra equilibration. Also available as a percentage in Simulation protocol."),
            ("min", "Hard minimum", "Blank disables the lower guard"),
            ("max", "Hard maximum", "Blank disables the upper guard"),
            ("max_abs_drift", "Maximum absolute drift", "Blank disables the drift guard"),
            ("hard_penalty", "Hard-limit penalty", "Added to the score if any guard fails"),
        ]
        for key, label, tooltip in definitions:
            field = QtWidgets.QLineEdit()
            value = settings.get(key)
            field.setText("" if value is None else str(value))
            field.setToolTip(tooltip)
            self.fields[key] = field
            form.addRow(label, field)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self):
        values = {}
        try:
            for key, field in self.fields.items():
                text = field.text().strip()
                values[key] = None if key in self.OPTIONAL_FIELDS and not text else float(text)
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Invalid value", "All populated settings must be numeric.")
            return
        for key in ("weight", "tail_scale", "drift_scale", "std_scale", "slope_scale", "hard_penalty"):
            if values[key] is None or values[key] <= 0.0:
                QtWidgets.QMessageBox.critical(self, "Invalid value", f"{key} must be positive.")
                return
        if not 0.0 <= values["skip_fraction"] < 1.0:
            QtWidgets.QMessageBox.critical(
                self, "Invalid value", "Initial fraction skipped must be at least 0 and less than 1."
            )
            return
        if values["min"] is not None and values["max"] is not None and values["min"] >= values["max"]:
            QtWidgets.QMessageBox.critical(self, "Invalid limits", "Hard minimum must be below hard maximum.")
            return
        if values["max_abs_drift"] is not None and values["max_abs_drift"] <= 0.0:
            QtWidgets.QMessageBox.critical(self, "Invalid limit", "Maximum absolute drift must be positive.")
            return
        self.settings.update(values)
        self.accept()


class OptimizePage(PageBase):
    settingsChanged = QtCore.Signal()
    continueRequested = QtCore.Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(
            "7 · Refine (optional)",
            "Optionally optimize property objectives with CG simulation trials. Next: Validate the best result, or skip with an existing CG trajectory.",
            parent,
        )
        box = QtWidgets.QGroupBox("Optimization settings")
        form = QtWidgets.QFormLayout(box)
        self.trials = QtWidgets.QSpinBox()
        self.trials.setRange(1, 100_000)
        self._protocol = {}
        self._trial_storage = {}
        self._search = {}
        self._optimization_root = None
        self._protocol_input = None
        trial_row = QtWidgets.QWidget()
        trial_layout = QtWidgets.QHBoxLayout(trial_row)
        trial_layout.setContentsMargins(0, 0, 0, 0)
        self.protocol_button = QtWidgets.QToolButton(text='Simulation protocol…')
        self.protocol_button.setToolTip('Ramp-up, production, Gaussian search, scoring and storage. Saved for this system; source LAMMPS files stay unchanged.')
        self.protocol_button.clicked.connect(self._edit_protocol)
        self.protocol_summary = StageDescription('Source input · ' + protocol_summary(DEFAULT_PROTOCOL))
        self.protocol_summary.setObjectName('pageDescription')
        trial_layout.addWidget(self.trials)
        self.budget_label = QtWidgets.QLabel()
        trial_layout.addWidget(self.budget_label)
        trial_layout.addWidget(self.protocol_button)
        self.results_button = QtWidgets.QToolButton(text='Trial results…')
        self.results_button.clicked.connect(self._show_trial_results)
        trial_layout.addWidget(self.results_button)
        self.continue_button = QtWidgets.QToolButton(text='Continue…')
        self.continue_button.setToolTip('Restore this system’s saved study and append more trials. Previous outputs are preserved.')
        self.continue_button.clicked.connect(self._continue_study)
        trial_layout.addWidget(self.continue_button)
        trial_layout.addWidget(self.protocol_summary, 1)
        self.command = QtWidgets.QLineEdit()
        self.command.setPlaceholderText("mpiexec -np 16 lmp -in input.data")
        self.structural = QtWidgets.QCheckBox("Include mapped-AA / CG distribution loss")
        form.addRow("Optuna trials", trial_row)
        form.addRow("LAMMPS command", self.command)
        form.addRow("Structural scoring", self.structural)
        property_box = QtWidgets.QGroupBox("Active property objectives")
        property_layout = QtWidgets.QVBoxLayout(property_box)
        explanation = QtWidgets.QLabel(
            "Add only the properties Optuna should score. Target and weight can be edited directly. "
            "Double-click a property, or use Configure, for statistical tolerances and hard limits."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("note")
        property_layout.addWidget(explanation)

        add_row = QtWidgets.QHBoxLayout()
        self.property_choice = QtWidgets.QComboBox()
        self.add_property_button = QtWidgets.QPushButton("＋ Add property")
        self.add_property_button.setObjectName("primaryButton")
        self.add_property_button.clicked.connect(self._add_selected_property)
        add_row.addWidget(self.property_choice, 1)
        add_row.addWidget(self.add_property_button)
        property_layout.addLayout(add_row)

        self.property_table = QtWidgets.QTableWidget(0, 6)
        self.property_table.setHorizontalHeaderLabels(
            ["Property", "Output column", "Target", "Weight", "Statistical tolerances", "Hard limits"]
        )
        self.property_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.property_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.property_table.setAlternatingRowColors(True)
        self.property_table.setMinimumHeight(160)
        self.property_table.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.property_table.verticalHeader().setVisible(False)
        self.property_table.cellDoubleClicked.connect(self._configure_row)
        header = self.property_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Stretch)
        property_layout.addWidget(self.property_table)

        action_row = QtWidgets.QHBoxLayout()
        self.configure_property_button = QtWidgets.QPushButton("Configure selected…")
        self.remove_property_button = QtWidgets.QPushButton("Remove selected")
        self.configure_property_button.clicked.connect(self._configure_selected)
        self.remove_property_button.clicked.connect(self._remove_selected)
        action_row.addWidget(self.configure_property_button)
        action_row.addWidget(self.remove_property_button)
        action_row.addStretch()
        property_layout.addLayout(action_row)
        self.layout.addWidget(property_box, 1)
        self.layout.addWidget(box)
        self._refresh_property_choices()
        self.add_run_row("optimize", "New study: Explore → best valid result → Refine. Continue… appends trials to the active study.")

    def _active_columns(self):
        return {
            self.property_table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)["column"]
            for row in range(self.property_table.rowCount())
        }

    def _refresh_property_choices(self):
        active = self._active_columns()
        self.property_choice.clear()
        for definition in PROPERTY_CATALOG:
            if definition["column"] not in active:
                self.property_choice.addItem(
                    f"{definition['label']}  ·  {definition['column']}", definition["column"]
                )
        enabled = self.property_choice.count() > 0
        self.property_choice.setEnabled(enabled)
        self.add_property_button.setEnabled(enabled)

    def _add_selected_property(self):
        column = self.property_choice.currentData()
        if not column:
            return
        self._append_property(property_settings(str(column)))
        self.property_table.selectRow(self.property_table.rowCount() - 1)
        self._refresh_property_choices()

    def _append_property(self, settings: dict):
        row = self.property_table.rowCount()
        self.property_table.insertRow(row)
        definition = property_definition(settings["column"])
        property_item = QtWidgets.QTableWidgetItem(definition["label"])
        property_item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(settings))
        column_item = QtWidgets.QTableWidgetItem(settings["column"])
        for item in (property_item, column_item):
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.property_table.setItem(row, 0, property_item)
        self.property_table.setItem(row, 1, column_item)
        self.property_table.setItem(
            row,
            2,
            QtWidgets.QTableWidgetItem(
                "" if settings.get("target") is None else str(settings["target"])
            ),
        )
        self.property_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(settings.get("weight", 1.0))))
        self.property_table.setItem(row, 4, QtWidgets.QTableWidgetItem(self._tolerance_summary(settings)))
        self.property_table.setItem(row, 5, QtWidgets.QTableWidgetItem(self._limit_summary(settings)))
        for column in (4, 5):
            item = self.property_table.item(row, column)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

    @staticmethod
    def _tolerance_summary(settings):
        return (
            f"tail {settings['tail_scale']:g} · drift {settings['drift_scale']:g} · "
            f"noise {settings['std_scale']:g} · slope {settings['slope_scale']:g} · "
            f"discard {100 * settings.get('skip_fraction', 0.):g}%"
        )

    @staticmethod
    def _limit_summary(settings):
        limits = []
        if settings.get("min") is not None:
            limits.append(f"min {settings['min']:g}")
        if settings.get("max") is not None:
            limits.append(f"max {settings['max']:g}")
        if settings.get("max_abs_drift") is not None:
            limits.append(f"|drift| {settings['max_abs_drift']:g}")
        return " · ".join(limits) if limits else "None"

    def _settings_for_row(self, row):
        settings = dict(self.property_table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole))
        target_text = self.property_table.item(row, 2).text().strip()
        try:
            settings["target"] = None if not target_text else float(target_text)
            settings["weight"] = float(self.property_table.item(row, 3).text())
        except ValueError as exc:
            raise ValueError(
                f"Invalid target or weight for {property_definition(settings['column'])['label']}."
            ) from exc
        if settings["weight"] <= 0.0:
            raise ValueError(
                f"Weight must be positive for {property_definition(settings['column'])['label']}."
            )
        return settings

    def _configure_selected(self):
        row = self.property_table.currentRow()
        if row < 0:
            return
        self._configure_row(row, 0)

    def _configure_row(self, row, _column):
        try:
            settings = self._settings_for_row(row)
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid objective", str(exc))
            return
        dialog = PropertySettingsDialog(settings, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        settings = dialog.settings
        self.property_table.item(row, 0).setData(QtCore.Qt.ItemDataRole.UserRole, settings)
        self.property_table.item(row, 2).setText(
            "" if settings.get("target") is None else str(settings["target"])
        )
        self.property_table.item(row, 3).setText(str(settings["weight"]))
        self.property_table.item(row, 4).setText(self._tolerance_summary(settings))
        self.property_table.item(row, 5).setText(self._limit_summary(settings))

    def _remove_selected(self):
        row = self.property_table.currentRow()
        if row < 0:
            return
        self.property_table.removeRow(row)
        self._refresh_property_choices()

    def _source_input_text(self):
        try:
            return self._protocol_input.read_text(encoding='utf-8') if self._protocol_input else None
        except OSError:
            return None

    def _refresh_protocol_summary(self):
        if self._protocol:
            source = self._source_input_text()
            try:
                base = read_protocol(source)[0] if source is not None else DEFAULT_PROTOCOL
            except ValueError:
                base = DEFAULT_PROTOCOL
            text = protocol_summary(merge_protocol(base, self._protocol))
        else:
            source = self._source_input_text()
            try:
                text = ('Source input · ' + protocol_summary(read_protocol(source)[0]) if source is not None
                        else 'Generated defaults · ' + protocol_summary(DEFAULT_PROTOCOL))
            except ValueError:
                text = 'Custom source input · unchanged'
        from .optimization_search import normalize_search
        search = normalize_search(self._search)
        if search['mode'] == 'incremental':
            text = 'Incremental · ' + text
        staged = search['mode'] == 'explore_refine'
        self.trials.setVisible(not staged)
        self.budget_label.setVisible(staged)
        self.budget_label.setText('Explore {} + Refine {}'.format(search['explore_trials'], search['refine_trials']))
        self.protocol_summary.setText(text)

    def _edit_protocol(self):
        try:
            properties = [self._settings_for_row(row) for row in range(self.property_table.rowCount())]
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Check property objectives', str(exc))
            return
        dialog = SimulationProtocolDialog(self._protocol, properties, self._source_input_text(), self,
                                          trial_storage=self._trial_storage, search=self._search)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._protocol = dict(dialog.overrides)
            self._trial_storage = dict(dialog.trial_storage)
            self._search = dict(dialog.search)
            self.property_table.setRowCount(0)
            for settings in dialog.properties:
                self._append_property(settings)
            self._refresh_property_choices()
            self._refresh_protocol_summary()
            self.settingsChanged.emit()
        dialog.deleteLater()

    def _show_trial_results(self):
        if self._optimization_root is None:
            return
        from .trial_results_dialog import TrialResultsDialog
        dialog = TrialResultsDialog(self._optimization_root, self)
        dialog.exec()
        dialog.deleteLater()

    def _continue_study(self):
        if self._optimization_root is None:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Continue study · additional trials')
        layout = QtWidgets.QVBoxLayout(dialog)
        row = QtWidgets.QHBoxLayout()
        explore, refine = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin in (explore, refine):
            spin.setRange(0, 100_000)
        from .optimization_history import best_directory
        if not best_directory(self._optimization_root).is_dir():
            explore.setValue(20)
        refine.setValue(20)
        for label, field in (('Add Explore', explore), ('Add Refine', refine)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(field)
        layout.addLayout(row)
        note = QtWidgets.QLabel('Resume saved history and best result. No second baseline; existing trials stay untouched.')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText('Continue')
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            if explore.value() + refine.value() < 1:
                QtWidgets.QMessageBox.warning(self, 'Add trials', 'Choose at least one additional trial.')
            else:
                self.continueRequested.emit(explore.value(), refine.value())
        dialog.deleteLater()

    def load_settings(self, execution, system=None, root=None):
        self.trials.setValue(execution.optimize_trials)
        self.command.setText(display_command(execution.lammps_command))
        self.structural.setChecked(execution.include_structural_loss)
        self._protocol = dict(system.optimization_protocol) if system is not None else {}
        self._trial_storage = dict(system.trial_storage) if system is not None else {}
        self._search = dict(system.optimization_search) if system is not None else {}
        self._optimization_root = Path(root) / 'optimization' if root is not None else None
        self.results_button.setEnabled(self._optimization_root is not None)
        self.continue_button.setEnabled(self._optimization_root is not None and self._optimization_root.is_dir())
        self._protocol_input = Path(root) / 'output' / 'lammps' / 'input.data' if root is not None else None
        self.protocol_button.setEnabled(system is not None)
        self._refresh_protocol_summary()
        self.property_table.setRowCount(0)
        if system is not None:
            for settings in system.optimization_properties:
                self._append_property(settings)
        self._refresh_property_choices()

    def apply_settings(self, execution, system=None):
        command = split_command(self.command.text().strip())
        if not command:
            raise ValueError("LAMMPS command cannot be empty.")
        execution.optimize_trials = self.trials.value()
        execution.lammps_command = command
        execution.include_structural_loss = self.structural.isChecked()
        if system is not None:
            selected = [self._settings_for_row(row) for row in range(self.property_table.rowCount())]
            system.optimization_properties = selected
            system.optimization_protocol = dict(self._protocol)
            system.trial_storage = dict(self._trial_storage)
            system.optimization_search = dict(self._search)


class MergePage(PageBase):
    def __init__(self, parent=None):
        super().__init__(
            "Combine systems",
            "Combine canonical interaction tables from two or more completed system optimizations, then optionally refine the transferable force field globally.",
            parent,
        )
        tabs = self.add_subtabs()
        systems_layout = self.add_subtab(tabs, "Systems")
        optimizer_layout = self.add_subtab(tabs, "Optimizer")

        self.systems = QtWidgets.QListWidget()
        self.systems.setMinimumHeight(120)
        self.optimize = QtWidgets.QCheckBox("Run global Optuna refinement after deterministic merge")
        self.trials = QtWidgets.QSpinBox()
        self.trials.setRange(1, 100_000)
        form_box = QtWidgets.QGroupBox("Merge inputs")
        form = QtWidgets.QVBoxLayout(form_box)
        form.addWidget(self.systems)
        systems_layout.addWidget(form_box)
        systems_layout.addStretch()

        optimizer_box = QtWidgets.QGroupBox("Global refinement")
        optimizer_form = QtWidgets.QFormLayout(optimizer_box)
        optimizer_form.addRow("Global optimizer", self.optimize)
        optimizer_form.addRow("Optuna trials", self.trials)
        optimizer_layout.addWidget(optimizer_box)
        optimizer_layout.addStretch()
        self.add_run_row("merge", "Outputs: merge/combined_forcefield and, when enabled, merge/best")

    def load_project(self, project):
        self.systems.clear()
        selected = set(project.merge_system_ids)
        for system in project.systems:
            item = QtWidgets.QListWidgetItem(system.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, system.id)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            checked = system.id in selected or (not selected and system.stages.get("optimize") == "complete")
            item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
            if system.stages.get("optimize") != "complete":
                item.setText(f"{system.name}  · optimization not marked complete")
            self.systems.addItem(item)
        self.optimize.setChecked(project.execution.merge_optimize)
        self.trials.setValue(project.execution.merge_trials)
        self.set_status(project.merge_status)

    def apply(self, project):
        project.merge_system_ids = [
            self.systems.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
            for index in range(self.systems.count())
            if self.systems.item(index).checkState() == QtCore.Qt.CheckState.Checked
        ]
        project.execution.merge_optimize = self.optimize.isChecked()
        project.execution.merge_trials = self.trials.value()





