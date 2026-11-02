"""On-demand diagnostics; no permanent panel or trajectory loading."""
from PySide6 import QtCore, QtWidgets

from .trial_diagnostics import load_trial_results, log_tail


class TrialResultsDialog(QtWidgets.QDialog):
    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Refine · Trial results')
        self.resize(1080, 520)
        layout = QtWidgets.QVBoxLayout(self)
        self.rows = load_trial_results(root)
        valid = sum(row['outcome'] == 'Valid' for row in self.rows)
        layout.addWidget(QtWidgets.QLabel('{} trials · {} valid · {} rejected'.format(
            len(self.rows), valid, len(self.rows) - valid)))
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.table = QtWidgets.QTableWidget(len(self.rows), 5)
        self.table.setHorizontalHeaderLabels(['Trial', 'Type', 'Score', 'Outcome', 'Reason'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText('Select a trial to inspect its reason and LAMMPS log.')
        split.addWidget(self.table)
        split.addWidget(self.details)
        split.setSizes([600, 480])
        layout.addWidget(split, 1)
        for index, row in enumerate(self.rows):
            score = row['score']
            values = [str(row['trial']), row['kind'], '' if score is None else '{:.6g}'.format(score),
                      row['outcome'], row['reason']]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(value + '\n' + str(row['started']) + '\n' + row['study'])
                self.table.setItem(index, column, item)
        self.table.currentCellChanged.connect(self._select)
        if self.rows:
            self.table.selectRow(len(self.rows) - 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select(self, row, *_):
        if not 0 <= row < len(self.rows):
            return
        result = self.rows[row]
        details = '{} · {}\n{}\nStudy: {}\nStarted: {}\nRecord: {}'.format(
            result['kind'], result['outcome'], result['reason'], result['study'], result['started'], result['directory'])
        if result['changed']:
            details += '\nReference: {} · change: {} / {}'.format(
                result['reference'], result['changed'], result['parameter'])
        self.details.setPlainText(details + '\n\n' + log_tail(result['directory'] / 'optuna_stdout.log'))
