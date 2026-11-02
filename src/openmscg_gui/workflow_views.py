"""Presentation-only grouping of the existing scientific stages."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from PySide6 import QtCore, QtWidgets
from .pages import PageBase
from .results_viewers import PlotPanel
from .fit_settings import (validate_override, automatic_defaults, normalize_override,
                           resolve_override, histogram_samples)


class WorkflowGroup(QtWidgets.QWidget):
    def __init__(self, pages, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName('stageSubTabs')
        self.tabs.setDocumentMode(True)
        for title, page in pages:
            self.tabs.addTab(page, title)
        layout.addWidget(self.tabs)


class FitValueSpinBox(QtWidgets.QDoubleSpinBox):
    def textFromValue(self, value):
        # Retain eight-decimal precision without displaying a wall of zeros.
        return super().textFromValue(value).rstrip('0').rstrip(self.locale().decimalPoint())


class DistributionPage(PageBase):
    continueRequested = QtCore.Signal()
    overrideChanged = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__('3 · Configure fit',
            'Review histogram bounds, spacing and spline order. Defaults are ready to use; continue to Fit when ready.',
            parent, scrollable=False)
        self.root = self._key = None
        self.entries = []
        self.overrides = {}
        self._entry = None
        self._axis = None
        self._loading = False
        self._drag = None
        self._overlays = []
        self._x = []
        self._samples = []
        self._auto_dirty = False
        self._original_bounds = True
        row = QtWidgets.QHBoxLayout()
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(['All interactions', 'Pair', 'Bond', 'Angle', 'Dihedral'])
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText('Find a bead or interaction…')
        row.addWidget(self.kind)
        row.addWidget(self.filter, 1)
        self.layout.addLayout(row)
        split = QtWidgets.QSplitter()
        split.setChildrenCollapsible(False)
        self.interactions = QtWidgets.QListWidget()
        self.interactions.setMinimumWidth(180)
        self.plot = PlotPanel()
        split.addWidget(self.interactions)
        split.addWidget(self.plot)
        split.setStretchFactor(1, 1)
        split.setSizes([250, 850])
        self.layout.addWidget(split, 1)

        self.editor = QtWidgets.QWidget()
        controls = QtWidgets.QHBoxLayout(self.editor)
        controls.setContentsMargins(0, 0, 0, 0)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Automatic', 'Manual'])
        self.mode.setToolTip('Bounds only: Automatic derives them from the histogram; Manual lets you place them. Spline spacing and order are independent. Reset restores all original settings.')
        controls.addWidget(self.mode)
        self.control_stack = QtWidgets.QStackedWidget()
        automatic_panel = QtWidgets.QWidget()
        automatic_layout = QtWidgets.QHBoxLayout(automatic_panel)
        automatic_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_method = QtWidgets.QComboBox()
        self.auto_method.addItem('Trim tails', 'quantile')
        self.auto_method.addItem('Relative peak height', 'threshold')
        self.auto_method.setToolTip('Trim tails uses cumulative histogram weight, not sample counts. Peak height selects the first and last bins above a fraction of the peak.')
        automatic_layout.addWidget(self.auto_method)
        self.auto_values = QtWidgets.QStackedWidget()
        tails = QtWidgets.QWidget()
        tails_layout = QtWidgets.QHBoxLayout(tails)
        tails_layout.setContentsMargins(0, 0, 0, 0)
        self.tail_lower = FitValueSpinBox()
        self.tail_upper = FitValueSpinBox()
        self.peak_threshold = FitValueSpinBox()
        for widget in (self.tail_lower, self.tail_upper, self.peak_threshold):
            widget.setDecimals(4)
            widget.setRange(0, 100)
            widget.setSuffix('%')
            widget.setSingleStep(.5)
            widget.setKeyboardTracking(False)
            widget.valueChanged.connect(self._automatic_changed)
            widget.editingFinished.connect(self._commit)
        self.peak_threshold.setMaximum(99.9999)
        self.tail_lower.setToolTip('Lower cumulative histogram-weight percentile. Default: 0.5%.')
        self.tail_upper.setToolTip('Upper cumulative histogram-weight percentile. Default: 99.5%.')
        self.peak_threshold.setToolTip('Use the first and last bins strictly above this percentage of the highest bin. Default: 5%.')
        for title, widget in [('Lower %', self.tail_lower), ('Upper %', self.tail_upper)]:
            tails_layout.addWidget(QtWidgets.QLabel(title))
            tails_layout.addWidget(widget)
        tails_layout.addStretch()
        peak = QtWidgets.QWidget()
        peak_layout = QtWidgets.QHBoxLayout(peak)
        peak_layout.setContentsMargins(0, 0, 0, 0)
        peak_layout.addWidget(QtWidgets.QLabel('Threshold (% of peak)'))
        peak_layout.addWidget(self.peak_threshold)
        peak_layout.addStretch()
        self.auto_values.addWidget(tails)
        self.auto_values.addWidget(peak)
        automatic_layout.addWidget(self.auto_values, 1)
        self.control_stack.addWidget(automatic_panel)
        manual_panel = QtWidgets.QWidget()
        manual_layout = QtWidgets.QHBoxLayout(manual_panel)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        self.lower = FitValueSpinBox()
        self.upper = FitValueSpinBox()
        self.spacing = FitValueSpinBox()
        for title, widget in [('From', self.lower), ('To', self.upper), ('Spline spacing', self.spacing)]:
            widget.setDecimals(8)
            widget.setRange(-1e9, 1e9)
            widget.setKeyboardTracking(False)
            widget.setSingleStep(0.01)
            if widget is not self.spacing:
                manual_layout.addWidget(QtWidgets.QLabel(title))
                manual_layout.addWidget(widget)
                widget.valueChanged.connect(self._preview)
            widget.editingFinished.connect(self._commit)
        self.spacing.setMinimum(0.00000001)
        self.spacing.setToolTip('Smaller spacing means a finer spline basis, not more histogram samples. Table export uses this spacing too.')
        self.control_stack.addWidget(manual_panel)
        controls.addWidget(self.control_stack, 1)
        self.spacing_mode = QtWidgets.QComboBox()
        self.spacing_mode.addItems(['Default spacing', 'Custom spacing'])
        self.spacing_mode.setToolTip('Default preserves the existing spacing policy. Custom spacing works with either automatic or manual bounds; it also sets exported table spacing.')
        controls.addWidget(self.spacing_mode)
        self.spacing.setMaximumWidth(125)
        controls.addWidget(self.spacing)
        self.order = QtWidgets.QSpinBox()
        self.order.setRange(2, 6)
        self.order.setKeyboardTracking(False)
        self.order.setMaximumWidth(65)
        self.order.setToolTip('OpenMSCG B-spline order (not polynomial degree). Original default: 6. Changing it requires running Fit again.')
        controls.addWidget(QtWidgets.QLabel('Order'))
        controls.addWidget(self.order)
        self.spacing.valueChanged.connect(self._spacing_changed)
        self.spacing_mode.currentIndexChanged.connect(self._spacing_mode_changed)
        self.order.valueChanged.connect(self._spline_changed)
        self.order.editingFinished.connect(self._commit)
        self.reset_button = QtWidgets.QToolButton(text='Reset')
        self.reset_button.setToolTip('Restore original bounds, automatic settings, spline spacing and order for this interaction.')
        self.reset_button.clicked.connect(self._reset)
        controls.addStretch()
        controls.addWidget(self.reset_button)
        self.layout.addWidget(self.editor)
        self.details = QtWidgets.QLabel()
        self.details.setObjectName('pageDescription')
        self.details.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        self.layout.addWidget(self.details)
        self.next_button = QtWidgets.QPushButton('Continue to Fit')
        self.next_button.setToolTip('Save fitting settings and open 4 · Fit. This does not start a calculation.')
        self.next_button.setObjectName('primaryButton')
        self.next_button.clicked.connect(self._continue)
        self.header_layout.addWidget(self.next_button)
        self.kind.currentIndexChanged.connect(self._filter)
        self.filter.textChanged.connect(self._filter)
        self.interactions.currentItemChanged.connect(self._plot)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.auto_method.currentIndexChanged.connect(self._automatic_method_changed)
        self.plot.canvas.mpl_connect('button_press_event', self._press)
        self.plot.canvas.mpl_connect('motion_notify_event', self._motion)
        self.plot.canvas.mpl_connect('button_release_event', self._release)
        self.clear()

    def load_system_root(self, root, current=False, overrides=None):
        root = Path(root)
        plan = root / 'output' / 'work' / 'fit_plan.json'
        stamp = plan.stat().st_mtime_ns if plan.is_file() else None
        saved = deepcopy(overrides or {})
        key = (str(root), stamp, current, json.dumps(saved, sort_keys=True))
        if key == self._key:
            return
        self._key, self.root = key, root
        self.overrides = saved
        self.entries = []
        error = None
        try:
            if plan.is_file() and current:
                entries = json.loads(plan.read_text(encoding='utf-8')).get('entries', [])
                if not isinstance(entries, list) or any(
                    not isinstance(entry, dict) or not isinstance(entry.get('fit_params'), dict)
                    or not isinstance(entry.get('name'), str) for entry in entries
                ):
                    raise ValueError('Invalid interaction entries in the fit plan.')
                self.entries = entries
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            error = str(exc)
        self.next_button.setEnabled(current and bool(self.entries))
        self._filter()
        if error:
            self.details.setText('Cannot read fit plan: ' + error)

    def clear(self):
        self.root = self._key = None
        self.entries = []
        self.overrides = {}
        self.next_button.setEnabled(False)
        self._filter()

    def _filter(self, *_):
        previous = self._entry.get('name') if self._entry else None
        self.interactions.blockSignals(True)
        self.interactions.clear()
        selected = None
        for entry in self.entries:
            name = entry['name']
            if self.kind.currentIndex() and entry.get('kind') != self.kind.currentText():
                continue
            if self.filter.text().casefold() not in name.casefold():
                continue
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(self._settings_label(name))
            self.interactions.addItem(item)
            if name == previous:
                selected = item
        if selected:
            self.interactions.setCurrentItem(selected)
        elif self.interactions.count():
            self.interactions.setCurrentRow(0)
        self.interactions.blockSignals(False)
        self._plot(self.interactions.currentItem())

    def _plot(self, item, *_):
        self._drag = None
        self._entry = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
        self._overlays = []
        self._x = []
        self._samples = []
        self._auto_dirty = False
        self._original_bounds = True
        self.plot.figure.clear()
        self._axis = self.plot.figure.add_subplot(111)
        self.editor.setEnabled(False)
        if self._entry is None:
            self._axis.set_axis_off()
            self.details.setText('No matching distributions.' if self.entries else
                                 'Build the mapped trajectory in Map to generate current distributions.')
            self.plot.toolbar.update()
            self.plot.canvas.draw_idle()
            return
        entry = self._entry
        try:
            pairs = histogram_samples(self.root, entry)
            self._samples = pairs
            self._x, y = zip(*pairs)
            params = dict(entry['fit_params'])
            settings = automatic_defaults(entry)
            override = self.overrides.get(entry['name'])
            self._original_bounds = override is None or override.get('original_bounds', False)
            saved_error = ''
            if override is not None:
                try:
                    normalized = normalize_override(entry, override)
                    if normalized.get('mode') == 'automatic':
                        settings = normalized
                    params.update(resolve_override(entry, override, samples=pairs))
                except ValueError as exc:
                    saved_error = 'Saved override is invalid: ' + str(exc) + ' Reset or edit it before fitting.'
            self._loading = True
            try:
                manual = override is not None and override.get('mode', 'manual') != 'automatic'
                self.mode.setCurrentIndex(int(manual))
                self.auto_method.setCurrentIndex(self.auto_method.findData(settings['method']))
                self.auto_values.setCurrentIndex(self.auto_method.currentIndex())
                self.tail_lower.setValue(float(settings['lower_quantile']) * 100)
                self.tail_upper.setValue(float(settings['upper_quantile']) * 100)
                self.peak_threshold.setValue(float(settings['histogram_threshold_percent']))
                self.lower.setValue(float(params['min']))
                self.upper.setValue(float(params['max']))
                self.spacing.setValue(float(params['resolution']))
                self.spacing_mode.setCurrentIndex(int(bool(override and
                    ('resolution' in override and override.get('spacing_mode') != 'default'))))
                self.order.setValue(int(params.get('order', 6)))
                suffix = '' if entry['kind'] in ('Pair', 'Bond') else '°'
                for widget in (self.lower, self.upper, self.spacing):
                    widget.setSuffix(suffix)
                    widget.setSingleStep(0.01 if entry['kind'] in ('Pair', 'Bond') else 0.5)
            finally:
                self._loading = False
            self._axis.plot(self._x, y, color='#168477', linewidth=1.8)
            self._axis.set_xlabel('Distance (input length units)' if entry['kind'] in ('Pair', 'Bond') else 'Angle (degrees)')
            self._axis.set_ylabel('OpenMSCG histogram value')
            self._axis.set_title(entry['name'])
            self._axis.grid(alpha=.15)
            editable = entry['kind'] in ('Pair', 'Bond', 'Angle', 'Dihedral')
            self.editor.setEnabled(editable)
            self._enable_fields()
            self._draw_selection(expand=True)
            if saved_error:
                self.details.setText(saved_error)
            elif not editable:
                self.details.setText('Unsupported interaction kind.')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.editor.setEnabled(False)
            self._x = []
            self._axis.text(.5, .5, 'Distribution unavailable', ha='center', va='center', transform=self._axis.transAxes)
            self.details.setText(str(exc))
        self.plot.toolbar.update()
        self.plot.canvas.draw_idle()

    def _values(self):
        return {'min': self.lower.value(), 'max': self.upper.value(), 'resolution': self.spacing.value(),
                'order': self.order.value()}

    def _settings_label(self, name):
        override = self.overrides.get(name)
        if override is None:
            return 'Original automatic settings'
        return 'Custom automatic settings' if override.get('mode') == 'automatic' else 'Manual fitting settings'

    def _automatic_settings(self):
        settings = {'mode': 'automatic', 'method': self.auto_method.currentData(),
                'lower_quantile': self.tail_lower.value() / 100,
                'upper_quantile': self.tail_upper.value() / 100,
                'histogram_threshold_percent': self.peak_threshold.value(), 'order': self.order.value()}
        if self._original_bounds:
            settings['original_bounds'] = True
        if self.spacing_mode.currentIndex() == 1:
            settings['resolution'] = self.spacing.value()
        return settings

    def _spacing_changed(self, *_):
        if self._loading:
            return
        self.spacing_mode.blockSignals(True)
        self.spacing_mode.setCurrentIndex(1)
        self.spacing_mode.blockSignals(False)
        self._spline_changed()

    def _spacing_mode_changed(self, *_):
        if self._loading or not self._entry:
            return
        if self.spacing_mode.currentIndex() == 0 and self.mode.currentIndex() == 1:
            self._loading = True
            self.spacing.setValue(float(self._entry['fit_params']['resolution']))
            self._loading = False
        self._spline_changed()
        self._commit()

    def _spline_changed(self, *_):
        if self._loading or not self._entry:
            return
        self._auto_dirty = True
        self._enable_fields()
        self.reset_button.setEnabled(True)
        if self.mode.currentIndex() == 0:
            try:
                self._update_automatic_bounds()
            except ValueError as exc:
                self.details.setText(str(exc) + ' Not saved.')
                return
        self._preview()

    def _update_automatic_bounds(self):
        settings = normalize_override(self._entry, self._automatic_settings())
        params = resolve_override(self._entry, settings, samples=self._samples)
        self._loading = True
        try:
            self.lower.setValue(params['min'])
            self.upper.setValue(params['max'])
            self.spacing.setValue(params['resolution'])
        finally:
            self._loading = False
        return settings

    def _automatic_changed(self, *_):
        if self._loading or not self._entry or self.mode.currentIndex() != 0:
            return
        self._auto_dirty = True
        self._original_bounds = False
        self.reset_button.setEnabled(True)
        try:
            self._update_automatic_bounds()
        except ValueError as exc:
            self.details.setText(str(exc) + ' Not saved; showing the last valid interval.')
            return
        self._draw_selection(expand=True)

    def _automatic_method_changed(self, *_):
        self.auto_values.setCurrentIndex(self.auto_method.currentIndex())
        if not self._loading:
            self._automatic_changed()
            self._commit()

    def _enable_fields(self):
        manual = self.mode.currentIndex() == 1
        self.control_stack.setCurrentIndex(int(manual))
        for widget in (self.lower, self.upper):
            widget.setEnabled(manual)
        self.spacing.setEnabled(self.spacing_mode.currentIndex() == 1)
        periodic = self._entry and self._entry['kind'] == 'Dihedral'
        self.mode.setEnabled(not periodic)
        self.control_stack.setEnabled(not periodic)
        self.reset_button.setEnabled(manual or bool(self._entry and self._entry['name'] in self.overrides))

    def _draw_selection(self, expand=False):
        if not self._entry or not self._x:
            return
        for artist in self._overlays:
            artist.remove()
        self._overlays = []
        lo, hi, spacing = (self._values()[key] for key in ('min', 'max', 'resolution'))
        if hi <= lo or spacing <= 0:
            return
        if expand:
            original = self._entry['fit_params']
            left = min(self._x[0], lo, float(original['min']))
            right = max(self._x[-1], hi, float(original['max']))
            pad = max((right - left) * .04, 1e-6)
            self._axis.set_xlim(left - pad, right + pad)
        left, right = self._axis.get_xlim()
        self._overlays.append(self._axis.axvspan(lo, hi, color='#168477', alpha=.12))
        if lo > left:
            self._overlays.append(self._axis.axvspan(left, lo, color='#677573', alpha=.10))
        if hi < right:
            self._overlays.append(self._axis.axvspan(hi, right, color='#677573', alpha=.10))
        legend_lines = []
        for i, value in enumerate((lo, hi)):
            line = self._axis.axvline(value, color='#247866', linewidth=2, picker=8,
                                     label='Active bounds' if i == 0 else '_nolegend_')
            self._overlays.append(line)
            if i == 0:
                legend_lines.append(line)
        self._default_lines = []
        for i, key in enumerate(('min', 'max')):
            line = self._axis.axvline(float(self._entry['fit_params'][key]), color='#64706e',
                                     linestyle='--', linewidth=1.5, zorder=4,
                                     label='Original automatic bounds' if i == 0 else '_nolegend_')
            self._overlays.append(line)
            self._default_lines.append(line)
            if i == 0:
                legend_lines.append(line)
        if self.mode.currentIndex() == 0 and self.auto_method.currentData() == 'threshold':
            threshold = max(value for _, value in self._samples) * self.peak_threshold.value() / 100
            line = self._axis.axhline(threshold, color='#a67324', linestyle=':', linewidth=1.5,
                                     label=f'{self.peak_threshold.value():g}% of peak')
            self._overlays.append(line)
            legend_lines.append(line)
        self._overlays.append(self._axis.legend(handles=legend_lines, loc='upper right', fontsize=8))
        count = max(1, int(math.ceil((hi - lo) / spacing)))
        stride = max(1, int(math.ceil(count / 120)))
        guides = [lo + i * spacing for i in range(0, count, stride)]
        self._overlays.append(self._axis.vlines(
            guides, 0, .035, transform=self._axis.get_xaxis_transform(), color='#247866', alpha=.6))
        mode = ('Fixed periodic bounds' if self._entry['kind'] == 'Dihedral' else
                'Manual' if self.mode.currentIndex() else
                'Original automatic bounds' if self._original_bounds else
                'Automatic: ' + self.auto_method.currentText())
        text = f"{mode}\n{lo:g} – {hi:g} · spacing {spacing:g} · order {self.order.value()}"
        self._overlays.append(self._axis.text(.02, .97, text, transform=self._axis.transAxes,
            va='top', fontsize=9, color='#173f38', bbox=dict(facecolor='white', alpha=.9, edgecolor='none')))
        self.details.setText(
            f"{len(self._x)} bins · gray dashed: original · green: active · ticks: spline spacing. "
            + ('Periodic bounds are fixed; spacing and order remain editable.' if self._entry['kind'] == 'Dihedral' else
               'Percentiles use histogram weight, not sample counts.' if self.mode.currentIndex() == 0
               and self.auto_method.currentData() == 'quantile' else
               'The first and last bins above the threshold define the interval.' if self.mode.currentIndex() == 0 else
               'Drag green bounds to adjust the interval.')
        )
        self.details.setToolTip('Edits save when you finish a field or release a boundary. Nothing runs automatically. '
                                'Automatic spacing follows the original interval-count setting (200 by default). '
                                'Custom spacing and order are independent of automatic bounds. Cutoffs remain unchanged. '
                                'Changes require a new Fit; this is not a force preview. Interior low-height bins are not individually removed. '
                                'Dense spacing guides are thinned for display; this is not a fitted force.')
        self.plot.canvas.draw_idle()

    def _preview(self, *_):
        if self._loading or not self._entry:
            return
        try:
            validate_override(self._entry, self._values())
        except ValueError as exc:
            self.details.setText(str(exc) + ' Not saved.')
            return
        self._draw_selection(expand=self._drag is None)

    def _save(self, value):
        name = self._entry['name']
        if value is None:
            if name not in self.overrides:
                return
            self.overrides.pop(name)
        else:
            if self.overrides.get(name) == value:
                return
            self.overrides[name] = dict(value)
        item = self.interactions.currentItem()
        if item:
            item.setToolTip(self._settings_label(name))
        self.overrideChanged.emit(name, value)

    def _commit(self):
        if self._loading or not self._entry or not self.editor.isEnabled():
            return True
        if self.mode.currentIndex() == 0 and not self._auto_dirty:
            return True
        try:
            values = (self._update_automatic_bounds() if self.mode.currentIndex() == 0
                      else validate_override(self._entry, self._values()))
            if self.mode.currentIndex() == 1:
                values['spacing_mode'] = 'custom' if self.spacing_mode.currentIndex() else 'default'
        except ValueError as exc:
            self._plot(self.interactions.currentItem())
            self.details.setText(str(exc) + ' Not saved; previous settings restored.')
            return False
        self._save(values)
        self._auto_dirty = False
        self._draw_selection()
        return True

    def _mode_changed(self, index):
        if self._loading or not self._entry:
            return
        self._auto_dirty = True
        self._enable_fields()
        if index == 0:
            self._original_bounds = False
            self._commit()
        else:
            self._draw_selection(expand=True)
            self._commit()

    def _reset(self):
        if not self._entry:
            return
        self._save(None)
        self._plot(self.interactions.currentItem())

    def _continue(self):
        if self._commit():
            self.continueRequested.emit()

    def _press(self, event):
        if (event.button != 1 or event.inaxes is not self._axis
                or not self.editor.isEnabled() or self.plot.toolbar.mode or event.x is None):
            return
        if self._entry['kind'] == 'Dihedral':
            return
        positions = [self._axis.transData.transform((value, 0))[0]
                     for value in (self.lower.value(), self.upper.value())]
        distances = [abs(event.x - value) for value in positions]
        if min(distances) > 10:
            return
        self._drag = distances.index(min(distances))
        self._auto_dirty = False
        self._loading = True
        self.mode.setCurrentIndex(1)
        self._loading = False
        self._enable_fields()

    def _motion(self, event):
        if self._drag is None or event.x is None or event.y is None or not self.editor.isEnabled():
            return
        value = self._axis.transData.inverted().transform((event.x, event.y))[0]
        if not math.isfinite(value):
            return
        lo, hi, spacing = self.lower.value(), self.upper.value(), self.spacing.value()
        if self._drag == 0:
            value = max(0., min(value, hi - spacing))
            self.lower.setValue(value)
        else:
            maximum = 180. if self._entry['kind'] == 'Angle' else (
                float(self._entry['fit_params']['cutoff']) if self._entry['kind'] == 'Pair' else 1e9)
            value = max(lo + spacing, min(value, maximum))
            self.upper.setValue(value)
        self._preview()

    def _release(self, event):
        if self._drag is not None:
            self._drag = None
            self._commit()
