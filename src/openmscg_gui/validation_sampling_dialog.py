"""Compact structural-validation window editor."""
from PySide6 import QtWidgets

from .validation_sampling import DEFAULT_SAMPLING, MAXIMUM_FRAMES, normalize_sampling


class ValidationSamplingDialog(QtWidgets.QDialog):
    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Sampling · Validate')
        self.resize(600, 170)
        root = QtWidgets.QVBoxLayout(self)
        fields = QtWidgets.QHBoxLayout()
        root.addLayout(fields)
        self.maximum_frames = QtWidgets.QSpinBox()
        self.maximum_frames.setRange(2, MAXIMUM_FRAMES)
        self.maximum_frames.setSingleStep(25)
        self.maximum_frames.setToolTip('Maximum frames per trajectory. Shorter trajectories use every remaining frame.')
        self.discard_percent = QtWidgets.QDoubleSpinBox()
        self.discard_percent.setRange(0, 95)
        self.discard_percent.setDecimals(1)
        self.discard_percent.setSingleStep(5)
        self.discard_percent.setSuffix(' %')
        self.discard_percent.setToolTip('Skip this initial fraction of frames in each trajectory, rounded down to whole frames. No automatic equilibration detection.')
        for label, widget in (('Frames per trajectory', self.maximum_frames),
                              ('Discard initial', self.discard_percent)):
            form = QtWidgets.QFormLayout()
            form.addRow(label, widget)
            fields.addLayout(form, 1)
        note = QtWidgets.QLabel('Evenly spaced across the remaining AA and CG trajectories. Structural checks only; property-file averages are unchanged.')
        note.setObjectName('note')
        note.setWordWrap(True)
        root.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Reset |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel |
            QtWidgets.QDialogButtonBox.StandardButton.Apply)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Reset).clicked.connect(
            lambda: self.set_settings(DEFAULT_SAMPLING))
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.set_settings(settings)

    def set_settings(self, settings):
        settings = normalize_sampling(settings)
        self.maximum_frames.setValue(settings['maximum_frames'])
        self.discard_percent.setValue(settings['discard_percent'])

    def settings(self):
        return normalize_sampling(dict(maximum_frames=self.maximum_frames.value(),
                                       discard_percent=self.discard_percent.value()))
