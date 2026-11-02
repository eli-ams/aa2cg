"""Per-interaction table recipes; previews never export or replace tables."""
from copy import deepcopy
from threading import Event

from PySide6 import QtCore, QtWidgets

from .results_viewers import TableExplorer, read_table_rows, _interaction_x_label


class PreviewSignals(QtCore.QObject):
    finished = QtCore.Signal(int, object, str)


class PreviewJob(QtCore.QRunnable):
    def __init__(self, token, path, recipe):
        super().__init__()
        self.token, self.path, self.recipe = token, path, recipe
        self.signals = PreviewSignals()
        self.cancel = Event()

    def run(self):
        try:
            from .backend.table_processing import preview_table
            result = preview_table(self.path, self.recipe, self.cancel.is_set)
            self.signals.finished.emit(self.token, result, '')
        except Exception as exc:
            self.signals.finished.emit(self.token, None, str(exc))


class TableProcessingExplorer(TableExplorer):
    overrideChanged = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        self.overrides = {}
        self.outputs_current = False
        self._loading = True
        self._token = 0
        self._jobs = {}
        self._result = None
        self._raw = None
        self._pending_recipe = None
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(180)
        self._timer.timeout.connect(self._launch_preview)
        self.editor = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.editor)
        row.setContentsMargins(0, 0, 0, 0)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Automatic', 'Custom'])
        self.mode.setToolTip('Automatic keeps the existing first-accepted search. Custom uses one saved recipe. Neither mode exports tables until Prepare tables is run.')
        row.addWidget(self.mode)
        self.smooth = QtWidgets.QSpinBox()
        self.smooth.setRange(5, 1001)
        self.smooth.setSingleStep(2)
        self.smooth.setToolTip('Odd energy smoothing window. The existing force window remains 6 × this value + 1; polynomial orders are unchanged.')
        self.trim = QtWidgets.QDoubleSpinBox()
        self.trim.setRange(0, 49.99)
        self.trim.setDecimals(2)
        self.trim.setSuffix('%')
        self.trim.setToolTip('Fraction of table points trimmed from EACH end, not histogram populations.')
        self.left, self.right = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for widget in (self.left, self.right):
            widget.setRange(-180, 1000000)
            widget.setDecimals(4)
            widget.setToolTip('Extend toward this coordinate in table units; a target inside retained data does not trim it. Pair tables require a left extension. Damping and extension spacing stay unchanged.')
        self.fit_points = QtWidgets.QSpinBox()
        self.fit_points.setRange(2, 10000)
        self.fit_points.setToolTip('Points used in existing endpoint line fits. Bond/angle force tails retain their existing whole-range linear fit.')
        for label, widget in [('Smooth', self.smooth), ('Trim/end', self.trim),
                              ('Extend from', self.left), ('to', self.right), ('Fit points', self.fit_points)]:
            row.addWidget(QtWidgets.QLabel(label))
            widget.setKeyboardTracking(False)
            widget.setMaximumWidth(115)
            row.addWidget(widget)
            widget.editingFinished.connect(self._commit)
            widget.valueChanged.connect(self._preview_edit)
        row.addStretch()
        self.reset_button = QtWidgets.QToolButton(text='Reset')
        self.reset_button.setToolTip('Remove this interaction’s custom recipe and restore the original automatic search.')
        row.addWidget(self.reset_button)
        self.layout().insertWidget(2, self.editor)
        self.detail.setWordWrap(True)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.reset_button.clicked.connect(self._reset)
        self.source.insertItem(0, 'Processing preview', 'preview')
        self.source.setToolTip('Processing preview edits recipes without exporting. Other choices browse saved raw/processed outputs.')
        self.source.setCurrentIndex(0)
        self._loading = False
        self._select_row(-1)
        self.destroyed.connect(lambda *_: [job.cancel.set() for job in self._jobs.values()])

    def load_system_root(self, root, overrides=None, current=False):
        self.overrides = deepcopy(overrides or {})
        self.outputs_current = current
        super().load_system_root(root)

    def _discover(self):
        entries = super()._discover()
        return entries + [dict(e, status='preview') for e in entries if e['status'] == 'raw']

    def _select_row(self, index):
        self._token += 1
        for job in self._jobs.values():
            job.cancel.set()
        self._result = None
        self._raw = None
        self._pending_recipe = None
        if hasattr(self, '_timer'):
            self._timer.stop()
        super()._select_row(index)
        if not hasattr(self, 'editor'):
            return
        enabled = bool(self.selected_entry and self.selected_entry['status'] == 'preview' and self._raw is not None)
        self.editor.setEnabled(enabled)
        if enabled:
            from .backend.table_processing import default_processing_settings, normalize_processing_settings
            entry = self.selected_entry
            recipe = self.overrides.get(entry['canonical'])
            try:
                values = (normalize_processing_settings(entry['kind'], recipe, self._raw[0])
                          if recipe is not None else default_processing_settings(entry['kind']))
                self._load_controls(values, recipe is not None)
            except (ValueError, TypeError, KeyError) as exc:
                self._load_controls(default_processing_settings(entry['kind']), True)
                self.detail.setText('Saved recipe needs adjustment: ' + str(exc) + ' Edit it or Reset.')

    def _load_controls(self, recipe, custom):
        self._loading = True
        try:
            self.mode.setCurrentIndex(int(custom))
            self.smooth.setValue(recipe['smooth'])
            self.trim.setValue(0 if self.selected_entry['kind'] == 'Dihedral' else recipe['trim'] * 100)
            self.left.setValue(recipe['bounds'][0])
            self.right.setValue(recipe['bounds'][1])
            self.fit_points.setValue(recipe['fit_points'])
        finally:
            self._loading = False
        self._enable_controls()

    def _enable_controls(self):
        custom = self.mode.currentIndex() == 1
        periodic = self.selected_entry and self.selected_entry['kind'] == 'Dihedral'
        self.smooth.setEnabled(custom)
        for widget in (self.trim, self.left, self.right, self.fit_points):
            widget.setEnabled(custom and not periodic)
        self.reset_button.setEnabled(custom or bool(self.selected_entry and self.selected_entry['canonical'] in self.overrides))

    def _recipe(self):
        return dict(smooth=self.smooth.value(), trim=self.trim.value() / 100,
                    bounds=[self.left.value(), self.right.value()], fit_points=self.fit_points.value())

    def _mode_changed(self, *_):
        if self._loading or not self.selected_entry:
            return
        if self.mode.currentIndex() == 0:
            self._reset()
        else:
            self._enable_controls()
            from .backend.table_processing import normalize_processing_settings
            try:
                normalize_processing_settings(self.selected_entry['kind'], self._recipe(), self._raw[0])
            except ValueError as exc:
                # Allow repairing an automatic recipe that cannot be used as a
                # custom recipe; do not trap the user in disabled Automatic fields.
                self.detail.setText(str(exc) + ' Adjust the custom settings before saving.')
                return
            self._commit()

    def _commit(self):
        if self._loading or not self.selected_entry or self.mode.currentIndex() != 1 or not self.editor.isEnabled():
            return
        from .backend.table_processing import normalize_processing_settings
        entry = self.selected_entry
        try:
            recipe = normalize_processing_settings(entry['kind'], self._recipe(), self._raw[0])
        except ValueError as exc:
            self._select_row(self.table_list.currentRow())
            self.detail.setText(str(exc) + ' Not saved; previous settings restored.')
            return
        name = entry['canonical']
        if self.overrides.get(name) != recipe:
            self.overrides[name] = recipe
            self.outputs_current = False
            self.overrideChanged.emit(name, recipe)
        self._select_row(self.table_list.currentRow())

    def _preview_edit(self, *_):
        if self._loading or not self.selected_entry or self.mode.currentIndex() != 1 or not self.editor.isEnabled():
            return
        from .backend.table_processing import normalize_processing_settings
        self._token += 1
        for job in self._jobs.values():
            job.cancel.set()
        self._timer.stop()
        try:
            self._pending_recipe = normalize_processing_settings(self.selected_entry['kind'], self._recipe(), self._raw[0])
        except ValueError as exc:
            self.detail.setText(str(exc) + ' Not saved; showing the last valid preview.')
            return
        self.detail.setText('Updating preview… Finish the field to save this recipe.')
        self._timer.start()

    def _reset(self):
        if not self.selected_entry:
            return
        name = self.selected_entry['canonical']
        if name in self.overrides:
            self.overrides.pop(name)
            self.outputs_current = False
            self.overrideChanged.emit(name, None)
        self._select_row(self.table_list.currentRow())

    def _plot_entry(self, axis, entry, index):
        if entry['status'] != 'preview':
            super()._plot_entry(axis, entry, index)
            if not self.outputs_current:
                self.detail.setText(self.detail.text() + ' · Saved output; Prepare tables to update it.')
            return
        try:
            self._raw = read_table_rows(entry['path'])
            self._draw_preview(None)
            self.detail.setText('Calculating processing preview… No output tables are being changed.')
            self._timer.start()
        except (OSError, ValueError) as exc:
            axis.text(.5, .5, 'Raw table unavailable', transform=axis.transAxes, ha='center')
            self.detail.setText(str(exc))

    def _launch_preview(self):
        if not self.selected_entry or self.selected_entry['status'] != 'preview':
            return
        entry = self.selected_entry
        recipe = self._pending_recipe if self._pending_recipe is not None else self.overrides.get(entry['canonical'])
        job = PreviewJob(self._token, entry['path'], deepcopy(recipe))
        self._jobs[self._token] = job
        job.signals.finished.connect(self._preview_finished)
        self._pool.start(job)

    def _preview_finished(self, token, result, error):
        self._jobs.pop(token, None)
        if token != self._token:
            return  # A different table, system or recipe is already selected.
        if error:
            self.detail.setText('Preview unavailable: ' + error + ' Saved output tables were not changed.')
            return
        self._result = result
        self._draw_preview(result)
        if self._pending_recipe is not None:
            self.detail.setText(self.detail.text() + ' Finish the field to save.')
        if self.mode.currentIndex() == 0:
            self._load_controls(result['recipe'], False)

    def _draw_preview(self, result):
        self.plot.figure.clear()
        energy_axis = self.plot.figure.add_subplot(111)
        force_axis = energy_axis.twinx()
        raw = self._raw
        lines = energy_axis.plot(raw[0], raw[1], '--', color='#1769aa', alpha=.5, label='Raw energy')
        lines += force_axis.plot(raw[0], raw[2], '--', color='#d05a00', alpha=.5, label='Raw force')
        if result:
            x, energy, force = result['curves']
            lines += energy_axis.plot(x, energy, color='#1769aa', label='Preview energy')
            lines += force_axis.plot(x, force, color='#d05a00', label='Preview force')
            recipe = result['recipe']
            cut = 0 if self.selected_entry['kind'] == 'Dihedral' else int(len(raw[0]) * recipe['trim'])
            lo, hi = raw[0][cut], raw[0][len(raw[0]) - cut - 1]
            if cut:
                energy_axis.axvspan(raw[0][0], lo, color='#777777', alpha=.12)
                energy_axis.axvspan(hi, raw[0][-1], color='#777777', alpha=.12)
            for bound in (lo, hi):
                energy_axis.axvline(bound, color='#777777', linestyle=':', linewidth=1)
            if x[0] < lo:
                energy_axis.axvspan(x[0], lo, color='#168477', alpha=.08)
            if x[-1] > hi:
                energy_axis.axvspan(hi, x[-1], color='#168477', alpha=.08)
            reasons = {
                'ls+0': 'left-tail slope has the wrong sign', 'rs+0': 'right-tail slope has the wrong sign',
                'absls-0.05': 'left-tail slope is too small', 'absls-0.5': 'left-tail force slope is too small',
                'rs+0.2': 'right tail is not sufficiently flat', 'ls--': 'left-tail energies are negative',
                'ls<ctr': 'left-tail energy is below the interior maximum',
                'ls.rs+0': 'energy-tail slopes do not have the required signs',
                'ls.rs-0': 'force-tail slopes do not have the required signs',
                'noroot': 'required zero/threshold crossing was not found', 'nonfinite': 'nonfinite values',
            }
            reason = reasons.get(result['reason'], str(result['reason']) if result['reason'] else
                                 'existing energy/force boundary checks did not pass')
            status = 'Passes existing checks' if result['accepted'] else 'Rejected: ' + reason
            self.detail.setText(status + ' · Preview only; run Prepare tables to export. Gray: trimmed · green: extended.')
            self.detail.setToolTip('These are the existing acceptance heuristics, not a guarantee of simulation stability. '
                                   'Damping, polynomial orders, force construction and boundary checks are unchanged. '
                                   'Rejected previews are never promoted to accepted output.')
        energy_axis.set_title(self.selected_entry['canonical'] + ' · Processing preview')
        energy_axis.set_xlabel(_interaction_x_label(self.selected_entry['canonical']))
        energy_axis.set_ylabel('Potential energy (table units)', color='#1769aa')
        force_axis.set_ylabel('Force (table units)', color='#d05a00')
        energy_axis.grid(alpha=.18)
        energy_axis.legend(lines, [line.get_label() for line in lines], loc='best')
        self.plot.toolbar.update()
        self.plot.canvas.draw_idle()
