from __future__ import annotations

import json
from pathlib import Path
import re

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from .backend.validation_metrics import load_property_statistics
from .validation_sampling import normalize_sampling


def read_table_rows(path: str | Path) -> tuple[list[float], list[float], list[float]]:
    distances, potentials, forces = [], [], []
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 4:
                continue
            try:
                float(parts[0])
                x, potential, force = map(float, parts[1:4])
            except ValueError:
                continue
            distances.append(x)
            potentials.append(potential)
            forces.append(force)
    if not distances:
        raise ValueError(f"No four-column table data found in {path}")
    return distances, potentials, forces


class PlotPanel(QtWidgets.QWidget):
    def __init__(self, parent=None, show_toolbar: bool = True, figsize=(7.5, 5.5)):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=figsize, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self) if show_toolbar else None
        layout.addWidget(self.canvas, 1)
        if self.toolbar is not None:
            # Keep Matplotlib's navigation and export behavior, but expose its
            # original actions through one text-only menu below the plot.
            self.toolbar.hide()
            row = QtWidgets.QHBoxLayout()
            self.plot_mode = QtWidgets.QLabel('')
            self.plot_mode.setObjectName('pageDescription')
            row.addWidget(self.plot_mode, 1)
            self.plot_menu = QtWidgets.QMenu(self)
            for action in self.toolbar.actions():
                if isinstance(action, QtWidgets.QWidgetAction):
                    continue
                action.setIcon(QtGui.QIcon())
                action.setIconVisibleInMenu(False)
                self.plot_menu.addAction(action)
                if action.isCheckable():
                    action.toggled.connect(self._update_plot_mode)
            self.view_button = QtWidgets.QToolButton(text='Plot tools')
            self.view_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            self.view_button.setMenu(self.plot_menu)
            self.view_button.setToolTip('Reset view, history, pan, zoom, plot settings and save')
            row.addWidget(self.view_button)
            layout.addLayout(row)

    def _update_plot_mode(self, *_):
        active = [action.text() for action in self.plot_menu.actions()
                  if action.isCheckable() and action.isChecked()]
        self.plot_mode.setText(' · '.join(active))
        self.plot_mode.setToolTip('Toggle the active tool off in Plot tools to stop panning or zooming.')


class TableExplorer(QtWidgets.QWidget):
    """List-and-plot browser for raw and processed LAMMPS tables."""

    SOURCES = [
        ("Accepted", "accepted"),
        ("All processed", "processed"),
        ("Rejected", "rejected"),
        ("Raw", "raw"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.system_root = None
        self.entries = []
        self.filtered_entries = []
        self.selected_entry = None
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        controls = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        for label, value in self.SOURCES:
            self.source.addItem(label, value)
        self.source.setToolTip("Choose raw, accepted, processed or rejected tables")
        self.kind = QtWidgets.QComboBox()
        for label in ("All interactions", "Pair", "Bond", "Angle", "Dihedral"):
            self.kind.addItem(label, "" if label.startswith("All") else label)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Find a bead or interaction…")
        self.reload_button = QtWidgets.QToolButton(text="Reload")
        self.files_button = QtWidgets.QToolButton(text="Files")
        self.files_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        files_menu = QtWidgets.QMenu(self.files_button)
        self.open_action = files_menu.addAction("Open table file", self._open_selected)
        self.folder_action = files_menu.addAction("Open containing folder", self._open_folder)
        self.files_button.setMenu(files_menu)
        controls.addWidget(self.source)
        controls.addWidget(self.kind)
        controls.addWidget(self.filter, 1)
        controls.addWidget(self.reload_button)
        controls.addWidget(self.files_button)
        root.addLayout(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.table_list = QtWidgets.QListWidget()
        self.table_list.setMinimumWidth(180)
        self.table_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.plot = PlotPanel(figsize=(9, 6))
        splitter.addWidget(self.table_list)
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 850])
        root.addWidget(splitter, 1)
        self.detail = QtWidgets.QLabel()
        self.detail.setObjectName("pageDescription")
        self.detail.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.detail.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.detail)

        self.source.currentIndexChanged.connect(self._refilter)
        self.kind.currentIndexChanged.connect(self._refilter)
        self.filter.textChanged.connect(self._refilter)
        self.reload_button.clicked.connect(self.reload)
        self.table_list.currentRowChanged.connect(self._select_row)
        self._select_row(-1)

    def load_system_root(self, root):
        new_root = Path(root) if root else None
        if new_root != self.system_root:
            self.selected_entry = None
        self.system_root = new_root
        self.reload()

    def reload(self):
        self.entries = self._discover()
        selected_source = self.source.currentData()
        if self.entries and not any(entry["status"] == selected_source for entry in self.entries):
            for preferred in ("accepted", "processed", "raw", "rejected"):
                if any(entry["status"] == preferred for entry in self.entries):
                    self.source.blockSignals(True)
                    self.source.setCurrentIndex(self.source.findData(preferred))
                    self.source.blockSignals(False)
                    break
        self._refilter()

    def show_accepted(self):
        self.source.blockSignals(True)
        self.source.setCurrentIndex(self.source.findData("accepted"))
        self.source.blockSignals(False)
        self.reload()

    def _discover(self):
        if self.system_root is None:
            return []
        output = self.system_root / "output"
        locations = {
            "raw": [output / "tables"],
            "processed": [output / "processed_tables" / "1_all"],
            "accepted": [output / "processed_tables" / "2_accepted"],
            "rejected": [
                output / "processed_tables" / "2_rejected",
                output / "processed_tables" / "2_finally_rejected",
            ],
        }
        entries = []
        for status, directories in locations.items():
            for directory in directories:
                if not directory.is_dir():
                    continue
                for path in sorted(directory.iterdir()):
                    canonical = _canonical_table_name(path.name)
                    if (
                        not path.is_file()
                        or canonical is None
                        or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
                    ):
                        continue
                    entries.append({
                        "path": path, "status": status, "canonical": canonical,
                        "kind": canonical.split("_", 1)[0],
                    })
        return entries

    def _refilter(self, *_):
        old = self.selected_entry
        selected_source = self.source.currentData()
        selected_kind = self.kind.currentData()
        query = self.filter.text().strip().lower()
        self.filtered_entries = [
            entry for entry in self.entries
            if entry["status"] == selected_source
            and (not selected_kind or entry["kind"] == selected_kind)
            and (not query or query in entry["canonical"].lower())
        ]
        self.table_list.blockSignals(True)
        self.table_list.clear()
        for entry in self.filtered_entries:
            item = QtWidgets.QListWidgetItem(entry["canonical"])
            item.setToolTip(str(entry["path"]))
            self.table_list.addItem(item)
        index = -1
        if self.filtered_entries:
            index = next((i for i, entry in enumerate(self.filtered_entries)
                          if old and entry["path"] == old["path"]), -1)
            if index < 0:
                index = next((i for i, entry in enumerate(self.filtered_entries)
                              if old and entry["canonical"] == old["canonical"]), 0)
        self.table_list.setCurrentRow(index)
        self.table_list.blockSignals(False)
        self._select_row(index)

    def _select_row(self, index):
        self.selected_entry = (
            self.filtered_entries[index] if 0 <= index < len(self.filtered_entries) else None
        )
        self.open_action.setEnabled(self.selected_entry is not None)
        self.folder_action.setEnabled(self.system_root is not None)
        self.plot.figure.clear()
        axis = self.plot.figure.add_subplot(111)
        entry = self.selected_entry
        if entry is None:
            axis.text(0.5, 0.5, "No matching tables\nRun force matching or change the filters.",
                      ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            self.detail.setText("No matching table files are currently available.")
            self.detail.setToolTip("")
        else:
            self._plot_entry(axis, entry, index)
        # A new table must not inherit the previous table's pan/zoom history.
        self.plot.toolbar.update()
        self.plot.canvas.draw_idle()

    def _plot_entry(self, energy_axis, entry, index):
        path = entry["path"]
        canonical = entry["canonical"]
        try:
            selected = read_table_rows(path)
        except Exception as exc:
            energy_axis.text(0.5, 0.5, "Could not read table\n" + str(exc),
                             ha="center", va="center", transform=energy_axis.transAxes, wrap=True)
            energy_axis.set_axis_off()
            self.detail.setText("Could not read selected table. Choose another table.")
            self.detail.setToolTip(str(exc))
            return

        raw_path = self.system_root / "output" / "tables" / canonical
        raw = None
        overlay_note = ""
        overlay_error = ""
        if raw_path.is_file() and raw_path != path:
            try:
                raw = read_table_rows(raw_path)
                overlay_note = " · dashed: raw"
            except Exception as exc:
                overlay_note = " · raw overlay unavailable"
                overlay_error = str(exc)
        force_axis = energy_axis.twinx()
        energy_color, force_color = "#1769aa", "#d05a00"
        lines = []
        if raw is not None:
            lines += energy_axis.plot(raw[0], raw[1], color=energy_color,
                                      linestyle="--", alpha=0.55, label="Raw energy")
            lines += force_axis.plot(raw[0], raw[2], color=force_color,
                                     linestyle="--", alpha=0.55, label="Raw force")
        label = self.source.currentText()
        lines += energy_axis.plot(selected[0], selected[1], color=energy_color, label=label + " energy")
        lines += force_axis.plot(selected[0], selected[2], color=force_color, label=label + " force")
        energy_axis.set_xlabel(_interaction_x_label(canonical))
        energy_axis.set_ylabel("Potential energy (table units)", color=energy_color)
        force_axis.set_ylabel("Force (table units)", color=force_color)
        energy_axis.tick_params(axis="y", colors=energy_color)
        force_axis.tick_params(axis="y", colors=force_color)
        energy_axis.grid(alpha=0.18)
        energy_axis.set_title(canonical)
        energy_axis.legend(lines, [line.get_label() for line in lines], loc="best")
        self.detail.setText(
            f"{index + 1}/{len(self.filtered_entries)} · {label} · "
            f"{len(selected[0])} points · blue: energy (left) · orange: force (right)" + overlay_note
        )
        self.detail.setToolTip(
            str(path) + "\nEnergy and force use separate Y scales and the same X coordinate."
            + (f"\nRaw overlay: {raw_path}" if raw is not None else "")
            + (f"\nRaw overlay could not be read: {overlay_error}" if overlay_error else "")
        )

    def _open_selected(self):
        if self.selected_entry:
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(self.selected_entry["path"].resolve()))
            )

    def _open_folder(self):
        path = self.selected_entry["path"].parent if self.selected_entry else (
            self.system_root / "output" if self.system_root else None
        )
        if path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path.resolve())))


class ValidationViewer(QtWidgets.QWidget):
    aaPropertiesRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.system_root: Path | None = None
        self.aa_properties_file: Path | None = None
        self.metrics: dict = {}
        self.sampling_settings = normalize_sampling()
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        action_row = QtWidgets.QHBoxLayout()
        self.sampling_button = QtWidgets.QPushButton('Sampling…')
        self.reload_button = QtWidgets.QPushButton("Reload results")
        self.open_aa_button = QtWidgets.QPushButton("Open mapped-AA trajectory")
        self.open_cg_button = QtWidgets.QPushButton("Open CG trajectory")
        self.folder_button = QtWidgets.QPushButton("Open validation folder")
        action_row.addWidget(self.sampling_button)
        action_row.addStretch()
        action_row.addWidget(self.reload_button)
        action_row.addWidget(self.open_aa_button)
        action_row.addWidget(self.open_cg_button)
        action_row.addWidget(self.folder_button)
        root.addLayout(action_row)

        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("stageSubTabs")
        tabs.setDocumentMode(True)
        self.tabs = tabs
        root.addWidget(tabs, 1)

        properties_tab = QtWidgets.QWidget()
        properties_layout = QtWidgets.QVBoxLayout(properties_tab)
        properties_layout.setContentsMargins(12, 12, 12, 12)
        tabs.addTab(properties_tab, "Properties")

        plot_tab = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_tab)
        plot_layout.setContentsMargins(12, 12, 12, 12)
        tabs.addTab(plot_tab, "Plots")

        self.view_choice = QtWidgets.QComboBox()
        self.view_choice.addItem("Intermolecular RDF", "rdf")
        self.view_choice.addItem("Bond lengths", "bonds")
        self.view_choice.addItem("Angles", "angles")
        plot_choice_row = QtWidgets.QHBoxLayout()
        plot_choice_row.addWidget(QtWidgets.QLabel("Plot"))
        plot_choice_row.addWidget(self.view_choice, 1)
        plot_layout.addLayout(plot_choice_row)

        self.scope = QtWidgets.QLabel("Run validation after a CG trajectory is available.")
        self.scope.setObjectName("note")
        self.scope.setWordWrap(True)
        self.scope.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        properties_layout.addWidget(self.scope)

        aa_source_row = QtWidgets.QHBoxLayout()
        self.aa_properties_status = QtWidgets.QLabel(
            "AA properties are not available. Load a LAMMPS ave/time-style output file to compare them."
        )
        self.aa_properties_status.setWordWrap(True)
        self.aa_properties_status.setObjectName("note")
        self.load_aa_properties_button = QtWidgets.QPushButton("Load AA properties…")
        self.load_aa_properties_button.setObjectName("primaryButton")
        aa_source_row.addWidget(self.aa_properties_status, 1)
        aa_source_row.addWidget(self.load_aa_properties_button)
        properties_layout.addLayout(aa_source_row)

        summary_box = QtWidgets.QGroupBox(
            "Structural agreement — 0 means matching distributions"
        )
        summary = QtWidgets.QGridLayout(summary_box)
        self.summary_labels = {}
        definitions = [
            ("rdf_distance", "RDF distance"),
            ("bond_distance", "Bond distance"),
            ("angle_distance", "Angle distance"),
            ("volume_relative_difference", "Volume difference"),
            ("number_density_relative_difference", "Number-density difference"),
        ]
        for index, (key, label) in enumerate(definitions):
            value = QtWidgets.QLabel("—")
            value.setObjectName("metricValue")
            summary.addWidget(QtWidgets.QLabel(label), 0, index)
            summary.addWidget(value, 1, index)
            self.summary_labels[key] = value
        properties_layout.addWidget(summary_box)

        comparison = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        comparison.setChildrenCollapsible(False)
        aa_box = QtWidgets.QGroupBox("Mapped AA reference")
        aa_layout = QtWidgets.QVBoxLayout(aa_box)
        cg_box = QtWidgets.QGroupBox("Coarse-grained result")
        cg_layout = QtWidgets.QVBoxLayout(cg_box)
        self.aa_properties_table = self._comparison_table()
        self.cg_properties_table = self._comparison_table()
        aa_layout.addWidget(self.aa_properties_table)
        cg_layout.addWidget(self.cg_properties_table)
        comparison.addWidget(aa_box)
        comparison.addWidget(cg_box)
        comparison.setSizes([500, 500])
        properties_layout.addWidget(comparison, 1)

        self.property_note = QtWidgets.QLabel(
            "All property-file records: mean ± standard deviation."
        )
        self.property_note.setToolTip('Property averages use all records in each property file. Sampling settings affect trajectory-based structural checks only.')
        self.property_note.setObjectName("note")
        self.property_note.setWordWrap(True)
        properties_layout.addWidget(self.property_note)

        self.plot = PlotPanel()
        self.plot.setMinimumHeight(380)
        plot_layout.addWidget(self.plot, 1)

        self.view_choice.currentIndexChanged.connect(self._plot_selected)
        self.reload_button.clicked.connect(self.reload)
        self.open_aa_button.clicked.connect(lambda: self._open_source("mapped_aa"))
        self.open_cg_button.clicked.connect(lambda: self._open_source("cg"))
        self.folder_button.clicked.connect(self._open_folder)
        self.load_aa_properties_button.clicked.connect(self.aaPropertiesRequested.emit)

    @staticmethod
    def _comparison_table() -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Property / quantity", "Mean ± std"])
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(32)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        table.setAlternatingRowColors(True)
        table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        return table

    def load_system(
        self,
        root: str | Path | None,
        aa_properties_file: str | Path | None = None,
        sampling_settings: dict | None = None,
    ):
        self.system_root = Path(root) if root else None
        self.aa_properties_file = Path(aa_properties_file) if aa_properties_file else None
        self.sampling_settings = normalize_sampling(sampling_settings)
        self.reload()

    def load_system_root(self, root: str | Path | None):
        self.load_system(root, self.aa_properties_file, self.sampling_settings)

    def set_aa_properties_file(self, path: str | Path | None):
        self.aa_properties_file = Path(path) if path else None
        self._refresh_properties()

    def cg_property_requests(self) -> list[dict]:
        return [
            {"label": item.get("label", item.get("column", "Property")), "column": item.get("column", "")}
            for item in self.metrics.get("cg_properties", [])
            if item.get("column")
        ]

    def reload(self):
        path = self.system_root / "validation" / "metrics.json" if self.system_root else None
        if path is None or not path.is_file():
            self.metrics = {}
            self.scope.setText("No validation metrics yet. Run this stage after optimization has produced a CG trajectory.")
            self.scope.setToolTip("")
        else:
            try:
                self.metrics = json.loads(path.read_text(encoding="utf-8"))
                sources = self.metrics.get("sources", {})
                sampling = self.metrics.get("sampling", {})
                uniform = sampling.get('method') == 'uniform_frame_indices'
                region = 'evenly spaced frames' if uniform else 'final frames (legacy)'
                discard = sampling.get('discard_percent', 0)
                stale = (not uniform or
                         sampling.get('maximum_frames_per_trajectory') != self.sampling_settings['maximum_frames'] or
                         discard != self.sampling_settings['discard_percent'])
                self.scope.setText(
                    ('Previous sampling · rerun validation.  ' if stale else '') +
                    f"{self.metrics.get('scope', 'AA / CG comparison')}  ·  "
                    f"{region}: AA {sampling.get('mapped_aa_frames', '—')} · "
                    f"CG {sampling.get('cg_frames', '—')}" +
                    (f" · initial discard {discard:g}%" if uniform else '')
                )
                windows = []
                for key, label in (('mapped_aa', 'AA'), ('cg', 'CG')):
                    window = sampling.get(key, {})
                    if window:
                        windows.append(f"{label}: {window.get('sampled_frames')} of {window.get('total_frames')} frames; "
                                       f"steps {window.get('first_timestep')}–{window.get('last_timestep')}; "
                                       f"discarded {window.get('discarded_frames')} frames")
                self.scope.setToolTip(
                    f"Mapped AA: {sources.get('mapped_aa', '—')}\n"
                    f"CG: {sources.get('cg', '—')}\n" + '\n'.join(windows)
                )
            except Exception as exc:
                self.metrics = {}
                self.scope.setText(f"Could not read {path}: {exc}")
                self.scope.setToolTip("")
        self._refresh_summary()
        self._refresh_properties()
        self._plot_selected()

    def _refresh_summary(self):
        summary = self.metrics.get("summary", {})
        for key, label in self.summary_labels.items():
            value = summary.get(key)
            suffix = "%" if key.endswith("relative_difference") and value is not None else ""
            shown = value * 100.0 if suffix else value
            label.setText("—" if shown is None else f"{shown:.3g}{suffix}")

    def _refresh_properties(self):
        cg_properties = self.metrics.get("cg_properties", []) if self.metrics else []
        requests = self.cg_property_requests()
        aa_properties = []
        error = ""
        if self.aa_properties_file is not None and requests:
            try:
                aa_properties = load_property_statistics(
                    self.aa_properties_file,
                    requests,
                )
            except (OSError, ValueError) as exc:
                error = str(exc)

        aa_by_column = {item["column"]: item for item in aa_properties}
        rows = []
        for cg_item in cg_properties:
            column = cg_item.get("column")
            aa_item = aa_by_column.get(column)
            if self.aa_properties_file is not None and aa_item is None:
                continue
            rows.append(
                {
                    "label": cg_item.get("label", column),
                    "mapped_aa": aa_item or {"mean": None, "std": None},
                    "cg": cg_item,
                }
            )
        for table in (self.aa_properties_table, self.cg_properties_table):
            table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            label = str(item.get("label", ""))
            for table, key in (
                (self.aa_properties_table, "mapped_aa"),
                (self.cg_properties_table, "cg"),
            ):
                table.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
                value = QtWidgets.QTableWidgetItem(_format_mean_std(item.get(key, {})))
                if item.get(key, {}).get("mean") is None:
                    value.setForeground(QtGui.QColor("#7b8582"))
                table.setItem(row, 1, value)

        self.load_aa_properties_button.setEnabled(bool(requests))
        self.load_aa_properties_button.setText(
            "Replace AA properties…" if self.aa_properties_file else "Load AA properties…"
        )
        if not requests:
            self.aa_properties_status.setText(
                "CG properties are not available yet. Complete optimization and run validation first."
            )
            self.aa_properties_status.setToolTip("")
            self.property_note.setText(
                "The comparison will include properties reported by the CG result."
            )
        elif self.aa_properties_file is None:
            self.aa_properties_status.setText(
                "AA properties are not available. Load a LAMMPS ave/time-style output file to compare them."
            )
            self.aa_properties_status.setToolTip("")
            self.property_note.setText(
                "The CG quantities are shown now. After loading AA data, only properties found "
                "in both files will remain in the side-by-side comparison."
            )
        elif error:
            self.aa_properties_status.setText(
                f"Could not read the selected AA properties file: {error}"
            )
            self.aa_properties_status.setToolTip(str(self.aa_properties_file))
            self.property_note.setText(
                "Choose another ave/time-style file containing named property columns."
            )
        elif not rows:
            self.aa_properties_status.setText(
                "The selected AA file contains none of the properties reported by the CG result."
            )
            self.aa_properties_status.setToolTip(str(self.aa_properties_file))
            self.property_note.setText(
                "Choose another file or include matching LAMMPS property column names."
            )
        else:
            self.aa_properties_status.setText(
                f"Loaded {len(rows)} matching AA propert{'y' if len(rows) == 1 else 'ies'} from "
                f"{self.aa_properties_file.name}."
            )
            self.aa_properties_status.setToolTip(str(self.aa_properties_file))
            self.property_note.setText(
                "All property-file records: mean ± standard deviation. Unmatched quantities are omitted."
            )

    def _plot_selected(self):
        self.plot.figure.clear()
        axis = self.plot.figure.add_subplot(111)
        key = self.view_choice.currentData()
        curve = self.metrics.get("curves", {}).get(key, {})
        x = curve.get("x", [])
        if x:
            axis.plot(x, curve.get("mapped_aa", []), label="Mapped AA", color="#1769aa")
            axis.plot(x, curve.get("cg", []), label="CG", color="#d05a00")
            axis.set_xlabel(curve.get("x_label", ""))
            axis.set_ylabel(curve.get("y_label", ""))
            axis.set_title(f"{self.view_choice.currentText()} · distance {_format_number(curve.get('distance'))}")
            axis.legend(loc="best")
            axis.grid(alpha=0.2)
        else:
            axis.text(0.5, 0.5, curve.get("detail", "No validation data yet"), ha="center", va="center")
            axis.set_axis_off()
        self.plot.canvas.draw_idle()

    def _open_source(self, key: str):
        value = self.metrics.get("sources", {}).get(key)
        if not value:
            return
        path = Path(value)
        if not path.is_absolute() and self.metrics.get('sources_relative_to') == 'project' and self.system_root:
            path = self.system_root.parent.parent / path
        if path.is_file():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path.resolve())))

    def _open_folder(self):
        if self.system_root:
            path = self.system_root / "validation"
            path.mkdir(parents=True, exist_ok=True)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path.resolve())))


def _canonical_table_name(name: str) -> str | None:
    match = re.match(r"(.+?\.table)(?:\..*)?$", name)
    return match.group(1) if match else None


def _interaction_x_label(name: str) -> str:
    kind = name.split("_", 1)[0]
    if kind == "Angle":
        return "Angle (degrees)"
    if kind == "Dihedral":
        return "Dihedral angle (degrees)"
    return "Distance (Å)"


def _format_number(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_mean_std(stats: dict) -> str:
    mean = stats.get("mean")
    if mean is None:
        return "Not available"
    standard_deviation = stats.get("std")
    if standard_deviation is None:
        return _format_number(mean)
    return f"{_format_number(mean)} ± {_format_number(standard_deviation)}"
