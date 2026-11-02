"""One compact editor for simulation duration, state point, and scoring windows."""
from copy import deepcopy

from PySide6 import QtCore, QtWidgets

from .optimization import property_definition
from .optimization_search import normalize_search
from .trial_storage import normalize_trial_storage
from .simulation_protocol import (DEFAULT_PROTOCOL, normalize_protocol, protocol_summary,
                                  read_protocol, prepare_trial_input, validate_scoring_window,
                                  ramp_stages, merge_protocol)


class CompactDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def textFromValue(self, value):
        return super().textFromValue(value).rstrip('0').rstrip(self.locale().decimalPoint())


class SimulationProtocolDialog(QtWidgets.QDialog):
    def __init__(self, overrides, properties, source_text=None, parent=None, trial_storage=None,
                 search=None):
        super().__init__(parent)
        self.setWindowTitle('Simulation protocol · Refine')
        self.resize(1120, 420)
        self.source_text = source_text
        self.overrides = deepcopy(overrides)
        self.properties = deepcopy(properties)
        self.trial_storage = normalize_trial_storage(trial_storage)
        self.search = normalize_search(search)
        self.base = dict(DEFAULT_PROTOCOL)
        self.source_supported = False
        if source_text is not None:
            try:
                self.base = read_protocol(source_text)[0]
                self.source_supported = True
            except ValueError:
                pass
        layout = QtWidgets.QVBoxLayout(self)
        self.custom = QtWidgets.QCheckBox('Customize protocol for this system')
        self.custom.setToolTip('Off: copy input.data unchanged. On: override only the simulation protocol in optimization trial copies.')
        layout.addWidget(self.custom)
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(16)
        layout.addLayout(columns, 1)
        self.ramp_box = QtWidgets.QGroupBox('Ramp-up · before production')
        ramp_form = QtWidgets.QFormLayout(self.ramp_box)
        ramp_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        ramp_form.setVerticalSpacing(8)
        ramp_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.ramp_enabled = QtWidgets.QCheckBox('Enable ramp-up')
        self.ramp_enabled.setToolTip('Temperature-controlled NVT startup at the production temperature, with a fixed box. No heating ramp; no scoring or trajectory output.')
        ramp_form.addRow(self.ramp_enabled)
        self.fields_box = QtWidgets.QGroupBox('Production · NPT')
        form = QtWidgets.QFormLayout(self.fields_box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(8)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.fields = {}
        for key, label, suffix in (
            ('ramp_steps', 'Ramp-up', ' steps'),
            ('equilibration_steps', 'Equilibration', ' steps'),
            ('production_steps', 'Sample', ' steps'),
            ('timestep_fs', 'Timestep', ' fs'),
            ('temperature_k', 'Temperature', ' K'),
            ('pressure_atm', 'Pressure', ' atm'),
        ):
            if key.endswith('_steps'):
                field = QtWidgets.QSpinBox()
                field.setRange(200 if key == 'production_steps' else 0, 2_000_000_000)
                field.setSingleStep(1000)
                field.setGroupSeparatorShown(True)
            else:
                field = CompactDoubleSpinBox()
                field.setDecimals(6)
                field.setRange(-1_000_000 if key == 'pressure_atm' else .000001, 1_000_000)
            field.setSuffix(suffix)
            field.setKeyboardTracking(False)
            field.setMinimumHeight(field.sizeHint().height())
            field.valueChanged.connect(self._update_summary)
            self.fields[key] = field
            (ramp_form if key in ('ramp_steps', 'equilibration_steps') else form).addRow(label, field)
        self.ramp_preview = QtWidgets.QLabel('')
        self.ramp_preview.setWordWrap(True)
        self.ramp_preview.setObjectName('note')
        ramp_form.addRow(self.ramp_preview)
        self.fields['ramp_steps'].setToolTip('Total ramp-up steps, split approximately 4:2:1 across quarter / half / full timestep for similar stage durations. At least 3 steps. The summary shows actual elapsed time; additional to production.')
        self.fields['equilibration_steps'].setToolTip('Optional additional NPT settling at the final timestep, after ramp-up. No samples; production starts at step zero afterward.')
        self.fields['timestep_fs'].setToolTip('Controls numerical integration, not convergence. A larger timestep can destabilize the simulation; increase production steps to sample longer.')
        columns.addWidget(self.ramp_box, 1)
        columns.addWidget(self.fields_box, 1)
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName('sectionTitle')
        self.summary.setWordWrap(True)

        scoring = QtWidgets.QGroupBox('Scoring window · production only')
        scoring_layout = QtWidgets.QVBoxLayout(scoring)
        scoring.setMinimumWidth(300)
        scoring.setToolTip('Discard excludes the beginning of production from each property’s score. It does not add simulation time. Equilibration is already excluded.')
        self.discard_table = QtWidgets.QTableWidget(len(properties), 2)
        self.discard_table.setMinimumHeight(160)
        self.discard_table.setHorizontalHeaderLabels(['Active property', 'Discard initial %'])
        self.discard_table.verticalHeader().hide()
        self.discard_table.setAlternatingRowColors(True)
        self.discard_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.discard_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.discard_fields = []
        for row, settings in enumerate(properties):
            item = QtWidgets.QTableWidgetItem(property_definition(settings['column'])['label'])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.discard_table.setItem(row, 0, item)
            spin = CompactDoubleSpinBox()
            spin.setDecimals(8)
            spin.setRange(0, 99.99999999)
            spin.setSuffix(' %')
            spin.setValue(100. * settings.get('skip_fraction', 0.))
            spin.setToolTip('The same per-property discard setting shown in Configure selected. A blank target uses the first retained production sample.')
            self.discard_fields.append(spin)
            self.discard_table.setCellWidget(row, 1, spin)
            self.discard_table.setRowHeight(row, max(
                self.discard_table.verticalHeader().defaultSectionSize(),
                spin.sizeHint().height() + 6))
        scoring_layout.addWidget(self.discard_table)
        if not properties:
            note = QtWidgets.QLabel('Add property objectives in Refine.')
            note.setObjectName('note')
            note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            scoring_layout.addWidget(note, 1)
            self.discard_table.hide()
        columns.addWidget(scoring, 1)
        search_row = QtWidgets.QHBoxLayout()
        search_row.addWidget(QtWidgets.QLabel('Search'))
        self.search_mode = QtWidgets.QComboBox()
        self.search_mode.addItem('Explore → Refine', 'explore_refine')
        self.search_mode.addItem('Current · broad search', 'current')
        self.search_mode.addItem('Conservative · incremental', 'incremental')
        self.search_mode.setCurrentIndex(self.search_mode.findData(self.search['mode']))
        self.search_mode.setToolTip('Explore → Refine: broad trials first, then small changes around their best valid result. Current and local-only searches remain available. Continue adds trials to a saved study instead of replacing it.')
        search_row.addWidget(self.search_mode)
        self.explore_label = QtWidgets.QLabel('Explore')
        self.refine_label = QtWidgets.QLabel('Refine')
        self.explore_trials = QtWidgets.QSpinBox()
        self.refine_trials = QtWidgets.QSpinBox()
        for spin, key in ((self.explore_trials, 'explore_trials'), (self.refine_trials, 'refine_trials')):
            spin.setRange(1 if key == 'explore_trials' else 0, 100_000)
            spin.setValue(self.search[key])
        self.explore_trials.setToolTip('Broad-search budget for a new study, including its optional baseline.')
        self.refine_trials.setToolTip('Additional local trials after Explore, starting from its best valid candidate. Zero skips refinement.')
        for widget in (self.explore_label, self.explore_trials, self.refine_label, self.refine_trials):
            search_row.addWidget(widget)
        self.search_step = CompactDoubleSpinBox()
        self.search_step.setDecimals(2)
        self.search_step.setRange(.01, 100.)
        self.search_step.setSuffix(' % step')
        self.search_step.setValue(self.search['step_percent'])
        self.search_step.setToolTip('Maximum change as a percentage of the existing parameter range. Amplitude, center and width are explored in turn; inactive Gaussians start with amplitude.')
        search_row.addWidget(self.search_step)
        self.baseline_first = QtWidgets.QCheckBox('Baseline first')
        self.baseline_first.setChecked(self.search['baseline_first'])
        self.baseline_first.setToolTip('One untouched-table trial, using the same protocol and scoring as all candidates. Counts inside the selected trial budget, not an extra run.')
        search_row.addWidget(self.baseline_first)
        search_row.addStretch(1)
        layout.addLayout(search_row)
        self.search_mode.currentIndexChanged.connect(self._update_search)
        self._update_search()
        storage_row = QtWidgets.QHBoxLayout()
        storage_row.addWidget(QtWidgets.QLabel('Trial storage'))
        self.storage_mode = QtWidgets.QComboBox()
        for label, mode in (('Keep everything', 'all'),
                            ('Keep improvement milestones', 'milestones'),
                            ('Keep only the best', 'best')):
            self.storage_mode.addItem(label, mode)
        self.storage_mode.setCurrentIndex(self.storage_mode.findData(self.trial_storage['mode']))
        self.storage_mode.setToolTip('Applies to future optimization runs, not existing results. '
            'Discarded run folders are removed after scoring, including failed trials and property-limit failures. '
            'Settings, scores, property summaries and logs are archived under optimization/trial_records. '
            'The current best is protected and exported to a versioned checkpoint. Continue never deletes previous checkpoints.')
        storage_row.addWidget(self.storage_mode)
        self.milestone_threshold = CompactDoubleSpinBox()
        self.milestone_threshold.setDecimals(2)
        self.milestone_threshold.setRange(0., 100.)
        self.milestone_threshold.setSuffix(' % improvement')
        self.milestone_threshold.setValue(self.trial_storage['threshold_percent'])
        self.milestone_threshold.setToolTip('Relative reduction in the minimized objective since the last '
            'retained milestone. Small improvements accumulate. The first valid trial and current '
            'best are always retained. Zero keeps every strict best-so-far improvement.')
        storage_row.addWidget(self.milestone_threshold)
        self.storage_warning = QtWidgets.QLabel('Other trial data permanently deleted.')
        self.storage_warning.setObjectName('note')
        self.storage_warning.setToolTip(self.storage_mode.toolTip())
        storage_row.addWidget(self.storage_warning)
        storage_row.addStretch(1)
        layout.addLayout(storage_row)
        self.storage_mode.currentIndexChanged.connect(self._update_storage)
        self._update_storage()
        self.summary.setToolTip('Sampling/averaging, damping and seed are unchanged. State-point changes do not change property targets. All trials use the same protocol; longer sampling does not guarantee convergence.')
        layout.addWidget(self.summary)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save |
                                           QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        reset = buttons.addButton('Use source input', QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        reset.setToolTip('Restore the source protocol; retain scoring windows, search and storage settings.')
        reset.clicked.connect(self._reset)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.custom.toggled.connect(self._update_summary)
        self.ramp_enabled.toggled.connect(self._toggle_ramp)
        self._load_fields(merge_protocol(self.base, overrides))
        self.custom.setChecked(bool(overrides))
        self._update_summary()

    def _load_fields(self, values):
        values = dict(values)
        values['ramp_steps'] = values.get('ramp_steps', 0) or (
            sum(steps for steps, _ in ramp_stages(values)) if values.get('ramp_ps', 0) else 0)
        for key, field in self.fields.items():
            field.blockSignals(True)
            # Clamp before crossing Qt's integer boundary: a legacy ps ramp can
            # contain more total steps than the new editor supports.
            field.setValue(min(values[key], field.maximum()))
            field.blockSignals(False)
        self.ramp_enabled.blockSignals(True)
        self.ramp_enabled.setChecked(values['ramp_steps'] > 0)
        self.ramp_enabled.blockSignals(False)

    def _toggle_ramp(self, enabled):
        field = self.fields['ramp_steps']
        if enabled and field.value() == 0:
            # Equivalent to the previous 10 ps suggestion at the default 1 fs.
            field.setValue(getattr(self, '_last_ramp_steps', 23335))
        elif not enabled:
            if field.value() > 0:
                self._last_ramp_steps = field.value()
            field.setValue(0)
        self._update_summary()

    def _values(self):
        return normalize_protocol({key: field.value() for key, field in self.fields.items()})

    def _update_summary(self, *_):
        self.fields_box.setEnabled(self.custom.isChecked())
        self.ramp_box.setEnabled(self.custom.isChecked())
        if not self.custom.isChecked():
            self._load_fields(self.base)
        if self.custom.isChecked():
            try:
                self.summary.setText(protocol_summary(self._values()))
            except ValueError as exc:
                self.summary.setText(str(exc))
        elif self.source_supported:
            self.summary.setText('Source input · ' + protocol_summary(self.base))
        elif self.source_text is None:
            self.summary.setText('Generated defaults · ' + protocol_summary(DEFAULT_PROTOCOL))
        else:
            self.summary.setText('Custom source input · unchanged')
        self.fields['ramp_steps'].setEnabled(self.ramp_enabled.isChecked())
        dt = self.fields['timestep_fs'].value()
        self.ramp_preview.setText('NVT · {:g} → {:g} → {:g} fs\nThen optional NPT settling; neither is scored.'
                                  .format(dt * .25, dt * .5, dt))

    def _reset(self):
        self.custom.setChecked(False)
        self._load_fields(self.base)
        self._update_summary()

    def _update_storage(self, *_):
        mode = self.storage_mode.currentData()
        self.milestone_threshold.setVisible(mode == 'milestones')
        self.storage_warning.setVisible(mode != 'all')

    def _update_search(self, *_):
        mode = self.search_mode.currentData()
        self.search_step.setVisible(mode in ('incremental', 'explore_refine'))
        for widget in (self.explore_label, self.explore_trials, self.refine_label, self.refine_trials):
            widget.setVisible(mode == 'explore_refine')

    def _accept(self):
        try:
            storage = normalize_trial_storage(dict(mode=self.storage_mode.currentData(),
                threshold_percent=self.milestone_threshold.value()))
            search = normalize_search(dict(mode=self.search_mode.currentData(),
                baseline_first=self.baseline_first.isChecked(), step_percent=self.search_step.value(),
                explore_trials=self.explore_trials.value(), refine_trials=self.refine_trials.value()))
            overrides = self._values() if self.custom.isChecked() else {}
            if self.custom.isChecked() and self.ramp_enabled.isChecked() and overrides.get('ramp_steps', 0) <= 0:
                raise ValueError('Set at least 3 ramp-up steps or switch ramp-up off.')
            properties = deepcopy(self.properties)
            for settings, spin in zip(properties, self.discard_fields):
                settings['skip_fraction'] = spin.value() / 100.
            if overrides:
                if self.source_text is not None:
                    prepare_trial_input(self.source_text, overrides)
                validate_scoring_window(overrides, properties)
            elif self.source_supported:
                validate_scoring_window(self.base, properties)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Check simulation protocol', str(exc))
            return
        self.overrides, self.properties = overrides, properties
        self.trial_storage = storage
        self.search = search
        self.accept()
