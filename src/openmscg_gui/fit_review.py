"""Read-only review of the requested fit and actual raw output tables."""
from copy import deepcopy
import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .fit_settings import resolve_override
from .results_viewers import PlotPanel, read_table_rows, _interaction_x_label


class FitReview(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = None
        self.entries = []
        self.overrides = {}
        self.status = 'pending'
        self.mapping_current = False
        self.method = 'independent'
        self.completed = {}
        self.requested = {}
        self.valid = True
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.notice = QtWidgets.QLabel('Build the mapped model to review fitting settings.')
        self.notice.setWordWrap(True)
        self.notice.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self.notice)
        split = QtWidgets.QSplitter()
        split.setChildrenCollapsible(False)
        self.settings = QtWidgets.QTreeWidget()
        self.settings.setHeaderLabels(['Interaction', 'Range', 'Spacing', 'Order', 'Settings'])
        self.settings.setRootIsDecorated(False)
        self.settings.setAlternatingRowColors(True)
        self.settings.setMinimumWidth(300)
        self.settings.setToolTip('Requested settings for the next run. Edit per interaction in Configure fit; Reset there restores its original settings.')
        self.plot = PlotPanel()
        split.addWidget(self.settings)
        split.addWidget(self.plot)
        split.setStretchFactor(1, 1)
        split.setSizes([480, 720])
        layout.addWidget(split, 1)
        self.detail = QtWidgets.QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.detail.setObjectName('pageDescription')
        layout.addWidget(self.detail)
        self.settings.currentItemChanged.connect(self._plot)
        self._plot()

    def load(self, root, overrides=None, mapping_current=False, status='pending', method='independent'):
        selected = self.settings.currentItem()
        previous = selected.text(0) if selected else None
        self.root = Path(root) if root is not None else None
        self.overrides = deepcopy(overrides or {})
        self.mapping_current, self.status, self.method = mapping_current, status, method
        self.entries, self.completed, self.requested = [], {}, {}
        self.valid = True
        error = ''
        if self.root:
            try:
                plan = self.root / 'output' / 'work' / 'fit_plan.json'
                if plan.is_file():
                    self.entries = json.loads(plan.read_text(encoding='utf-8'))['entries']
                    if not isinstance(self.entries, list) or any(
                            not isinstance(e, dict) or not isinstance(e.get('name'), str)
                            or not isinstance(e.get('fit_params'), dict) for e in self.entries):
                        raise ValueError('Invalid fitting plan entries.')
                record = self.root / 'output' / 'work' / 'force_match.json'
                if record.is_file():
                    self.completed = json.loads(record.read_text(encoding='utf-8'))
                    if not isinstance(self.completed, dict):
                        raise ValueError('Invalid completed-fit record.')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = 'Cannot read fit records: ' + str(exc)
                self.entries, self.completed = [], {}
                self.valid = False
        self.settings.blockSignals(True)
        self.settings.clear()
        chosen = None
        for entry in self.entries:
            try:
                params = dict(entry['fit_params'])
                override = self.overrides.get(entry['name'])
                if override is not None:
                    params.update(resolve_override(entry, override, root=self.root))
                self.requested[entry['name']] = params
                item = QtWidgets.QTreeWidgetItem([
                    entry['name'], '{:g} – {:g}'.format(params['min'], params['max']),
                    '{:g}'.format(params['resolution']), str(params['order']),
                    'Custom' if override is not None else 'Original',
                ])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (entry, params, ''))
                for column in range(5):
                    item.setToolTip(column, 'Model: {} · neighbor cutoff: {:g} (read-only)\n'
                                    'Bounds, spline spacing and order: edit in Configure fit.'.format(
                                        params.get('model', 'BSpline'), params['cutoff']))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.valid = False
                item = QtWidgets.QTreeWidgetItem([entry.get('name', '?'), 'Invalid', '', '', 'Fix settings'])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (entry, None, str(exc)))
                item.setToolTip(0, str(exc))
            self.settings.addTopLevelItem(item)
            if item.text(0) == previous:
                chosen = item
        for column in range(5):
            self.settings.resizeColumnToContents(column)
        self.settings.setCurrentItem(chosen or self.settings.topLevelItem(0))
        self.settings.blockSignals(False)
        self._plot()
        if error:
            self.notice.setText(error)

    def set_method(self, method):
        self.method = method
        self._plot()

    def set_status(self, status):
        self.status = status
        self._plot()

    def _is_current(self):
        snapshot_matches = True
        if 'fit_plan' in self.completed:
            try:
                snapshot_matches = {e['name']: e['fit_params'] for e in self.completed['fit_plan']['entries']} == self.requested
            except (KeyError, TypeError):
                snapshot_matches = False
        return (self.valid and bool(self.entries) and snapshot_matches
                and self.mapping_current and self.status == 'complete'
                and self.root is not None
                and all((self.root / 'output' / 'tables' / (e['name'] + '.table')).is_file()
                        for e in self.entries)
                and self.completed.get('method') == self.method
                and self.completed.get('fit_overrides') == self.overrides)

    def _plot(self, *_):
        self.plot.figure.clear()
        axis = self.plot.figure.add_subplot(111)
        item = self.settings.currentItem()
        current = self._is_current()
        self.notice.setText('Current raw fit · settings below match the completed run.' if current else
                            'Previous fit / not fitted — run Fit to produce results for the requested settings.'
                            if self.mapping_current else
                            'Mapping is not current — rebuild Map before fitting. Any curves below are previous results.')
        self.detail.setText('Settings are not a predicted force curve. Only actual raw table files are plotted.')
        self.detail.setToolTip('')
        if item is None or self.root is None:
            axis.text(.5, .5, 'No fitting plan available', ha='center', va='center', transform=axis.transAxes)
            axis.set_axis_off()
        else:
            entry, params, error = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            name = entry['name']
            path = self.root / 'output' / 'tables' / (name + '.table')
            try:
                x, energy, force = read_table_rows(path)
                force_axis = axis.twinx()
                energy_line = axis.plot(x, energy, color='#1769aa', label='Raw energy')
                force_line = force_axis.plot(x, force, color='#d05a00', label='Raw force')
                axis.set_xlabel(_interaction_x_label(name))
                axis.set_ylabel('Potential energy (table units)', color='#1769aa')
                force_axis.set_ylabel('Force (table units)', color='#d05a00')
                axis.grid(alpha=.18)
                axis.legend(energy_line + force_line, ['Raw energy', 'Raw force'], loc='best')
                axis.set_title(name + (' · Current raw fit' if current else ' · Previous / unverified output'))
                # A success-only snapshot, not effective_fit_plan.json (which is
                # written before a run and may belong to a failed attempt).
                fitted = next((e['fit_params'] for e in self.completed.get('fit_plan', {}).get('entries', [])
                               if e['name'] == name), None)
                detail = 'Blue: energy (left) · orange: force (right).'
                if fitted and self.status == 'complete':
                    detail += ' Last completed fit: {:g}–{:g}, spacing {:g}, order {}.'.format(
                        fitted['min'], fitted['max'], fitted['resolution'], fitted['order'])
                if not current:
                    detail += ' Rerun required; these curves do not preview edited settings.'
                if error:
                    detail += ' Invalid requested settings: ' + error
                self.detail.setText(detail)
                self.detail.setToolTip(str(path))
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                axis.text(.5, .5, 'No readable raw result\nRun force matching to generate this curve.',
                          ha='center', va='center', transform=axis.transAxes)
                axis.set_axis_off()
                self.detail.setText(error or 'No raw table available for this interaction.')
                self.detail.setToolTip(str(exc))
        self.plot.toolbar.update()
        self.plot.canvas.draw_idle()
