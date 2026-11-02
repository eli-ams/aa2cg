"""Compact export presets with cancellable, background ZIP creation."""
from copy import deepcopy
from pathlib import Path
import re
from threading import Event

from PySide6 import QtCore, QtWidgets

from .project_export import PRESETS, ExportCancelled, plan_export, write_export


def _size_text(size):
    value = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return '{:.2g} {}'.format(value, unit)
        value /= 1024


class ExportWorker(QtCore.QThread):
    progress = QtCore.Signal(int, str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.cancel = Event()
        self.value = None
        self.problem = ''
        self.was_cancelled = False

    def run(self):
        try:
            self.value = self.operation(self.cancel.is_set, self.progress.emit)
        except ExportCancelled:
            self.was_cancelled = True
        except Exception as exc:
            self.problem = str(exc)


class ProjectExportDialog(QtWidgets.QDialog):
    exported = QtCore.Signal(str)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = deepcopy(project)
        self.plan = None
        self.worker = None
        self.job = ''
        self.pending_close = False
        self.setWindowTitle('Export project')
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.resize(680, 235)
        root = QtWidgets.QVBoxLayout(self)
        choices = QtWidgets.QHBoxLayout()
        choices.addWidget(QtWidgets.QLabel('Include'))
        self.preset = QtWidgets.QComboBox()
        for key, label, description in PRESETS:
            self.preset.addItem(label, key)
            self.preset.setItemData(self.preset.count() - 1, description, QtCore.Qt.ItemDataRole.ToolTipRole)
        choices.addWidget(self.preset, 1)
        self.estimate = QtWidgets.QLabel('Checking files…')
        choices.addWidget(self.estimate)
        root.addLayout(choices)
        self.description = QtWidgets.QLabel()
        self.description.setWordWrap(True)
        self.description.setObjectName('note')
        root.addWidget(self.description)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)
        self.status = QtWidgets.QLabel('')
        self.status.setObjectName('note')
        root.addWidget(self.status)
        actions = QtWidgets.QHBoxLayout()
        note = QtWidgets.QLabel('Extract ZIP → Open project. Originals are kept.')
        note.setObjectName('note')
        actions.addWidget(note)
        actions.addStretch()
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.export_button = QtWidgets.QPushButton('Export ZIP…')
        self.export_button.setObjectName('primaryButton')
        self.export_button.setEnabled(False)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.export_button)
        root.addLayout(actions)
        self.cancel_button.clicked.connect(self.reject)
        self.export_button.clicked.connect(self._export)
        self.preset.currentIndexChanged.connect(self._plan)
        QtCore.QTimer.singleShot(0, self._plan)

    @property
    def busy(self):
        return self.worker is not None

    def _start(self, job, operation):
        self.job = job
        self.preset.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0 if job == 'plan' else 100)
        self.progress.setValue(0)
        self.worker = ExportWorker(operation, self)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _plan(self):
        if self.busy or self.pending_close:
            return
        self.plan = None
        preset = self.preset.currentData()
        self.description.setText(next(item[2] for item in PRESETS if item[0] == preset))
        self.estimate.setText('Checking files…')
        self.status.setText('Reading file sizes only; no simulations or trajectory parsing.')
        self._start('plan', lambda cancelled, progress: plan_export(self.project, preset, cancelled))

    def _export(self):
        if self.busy or self.plan is None:
            return
        name = re.sub(r'[^A-Za-z0-9_.-]+', '-', self.project.name).strip('-.') or 'project'
        initial = self.project.root.parent / ('{}-{}.zip'.format(name, self.plan.preset))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export portable project', str(initial), 'ZIP archives (*.zip)',
            options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        destination = Path(path if path.lower().endswith('.zip') else path + '.zip')
        if destination.exists() or destination.is_symlink():
            QtWidgets.QMessageBox.warning(self, 'Choose a new name', 'Existing archives are never overwritten.')
            return
        plan = self.plan
        self.status.setText('Compressing selected files…')
        self.cancel_button.setText('Cancel')
        self._start('write', lambda cancelled, progress: write_export(plan, destination, cancelled, progress))

    @QtCore.Slot(int, str)
    def _progress(self, percent, name):
        self.progress.setValue(percent)
        self.status.setText(self.fontMetrics().elidedText(name, QtCore.Qt.TextElideMode.ElideMiddle,
                                                        max(self.width() - 40, 100)))
        self.status.setToolTip(name)

    def _finished(self):
        worker = self.worker
        self.worker = None
        value, problem, cancelled = worker.value, worker.problem, worker.was_cancelled
        worker.deleteLater()
        if self.pending_close:
            if problem:
                QtWidgets.QMessageBox.warning(self, 'Export not completed', problem)
            super().reject()
            return
        self.preset.setEnabled(True)
        self.cancel_button.setEnabled(True)
        if problem:
            self.progress.hide()
            self.status.setText('Export not completed. Original project files are unchanged.')
            if self.job == 'plan':
                self.estimate.setText('Not ready')
            QtWidgets.QMessageBox.warning(self, 'Could not export project', problem)
        elif cancelled:
            self.progress.hide()
            self.status.setText('Export cancelled.')
        elif self.job == 'plan':
            self.plan = value
            self.progress.hide()
            self.estimate.setText('{} before compression'.format(_size_text(value.total_bytes)))
            self.status.setText('{} files · {} non-best trial files omitted'.format(
                len(value.entries), value.omitted_files))
            self.status.setToolTip('Size is the selected uncompressed data. ZIP size depends on compression. '
                                   'Checksums and software version metadata are added during export.')
        else:
            self.progress.setValue(100)
            self.status.setText('Exported {}'.format(Path(value).name))
            self.status.setToolTip(str(value))
            self.cancel_button.setText('Close')
            self.exported.emit(str(value))
        self.export_button.setEnabled(self.plan is not None)

    def reject(self):
        if self.busy:
            self.pending_close = True
            self.worker.cancel.set()
            self.cancel_button.setEnabled(False)
            self.status.setText('Cancelling export and removing temporary output…')
            return
        self.pending_close = True
        super().reject()

    def closeEvent(self, event):
        if self.busy:
            self.reject()
            event.ignore()
        else:
            self.pending_close = True
            super().closeEvent(event)
