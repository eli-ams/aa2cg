"""Compact linked AA/CG display; playback compares progress, not physical time."""
from PySide6 import QtCore, QtWidgets

from .trajectory_view import TrajectoryView


def relative_frame(position, maximum, frame_count):
    if maximum <= 0 or frame_count <= 1:
        return 0
    fraction = max(0., min(1., float(position) / maximum))
    return int(fraction * (frame_count - 1) + .5)


class TrajectoryComparison(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.aa_view = TrajectoryView(mapping=True)
        self.cg_view = TrajectoryView(cg=True)
        self.views = (self.aa_view, self.cg_view)
        self._syncing = False
        self._camera_state = None
        self._refit = True
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        for title, view in zip(('AA', 'Independent CG'), self.views):
            view.comparison_mode = True
            view.scope.setCurrentIndex(2)
            view.playback_bar.hide()
            panel = QtWidgets.QWidget()
            column = QtWidgets.QVBoxLayout(panel)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(2)
            heading = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(title)
            label.setObjectName('sectionTitle')
            heading.addWidget(label)
            heading.addStretch()
            heading.addWidget(view.frame_label)
            column.addLayout(heading)
            column.addWidget(view, 1)
            self.splitter.addWidget(panel)
            for control in (view.scope, view.species):
                control.currentIndexChanged.connect(lambda _, v=view: self._selection(v))
            view.instance.valueChanged.connect(lambda _, v=view: self._selection(v))
            for control in (view.dim, view.follow):
                control.toggled.connect(lambda _, v=view: self._selection(v))
            view.stateChanged.connect(self._state_changed)
            view.sceneChanged.connect(lambda reset, v=view: self._scene_changed(v, reset))
            if view.renderer:
                view.renderer.cameraChanged.connect(self._camera_changed)
        self.aa_view.representation.setCurrentIndex(0)
        self.splitter.setSizes([600, 600])
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        layout.addWidget(self.splitter, 1)
        # One strip replaces both independent playback bars. Selection and
        # display preferences remain in the existing View menu.
        row = QtWidgets.QHBoxLayout()
        self.play = QtWidgets.QToolButton(text='Play both')
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.progress = QtWidgets.QLabel('Relative progress · 0%')
        self.progress.setToolTip('The same fraction of each trajectory, not equal timesteps or matched physical motion. Each panel shows its own timestep.')
        self.view_button = QtWidgets.QToolButton(text='View ▾')
        self.view_button.setToolTip('Shared species, molecule and display options. Representation applies to AA only. Drag or zoom either panel to control both.')
        self.view_button.clicked.connect(lambda: self.aa_view._show_view_menu(self.view_button))
        for widget in (self.play, self.slider, self.progress, self.view_button):
            row.addWidget(widget, 1 if widget is self.slider else 0)
        layout.addLayout(row)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._advance)
        self.play.clicked.connect(self._toggle_play)
        self.slider.valueChanged.connect(self._seek)
        self.aa_view.representation.currentIndexChanged.connect(self._request_refit)
        self.aa_view.displayOptionsChanged.connect(self._display_options)
        self.aa_view.frameViewRequested.connect(self._frame_views)
        self._state_changed()

    def _ready(self):
        return all(v._model is not None and v._reader is not None and v._reader.frames
                   and not v._graphics_failure and not v._preview_error for v in self.views)

    def _idle(self):
        return all(not v._busy and not v._dirty and not v._debounce.isActive() for v in self.views)

    def _pause(self):
        self._timer.stop()
        self.play.setText('Play both')

    def _state_changed(self):
        ready = self._ready()
        self.play.setEnabled(bool(ready))
        self.slider.setEnabled(bool(ready))
        if not ready:
            self._pause()
            self.slider.setRange(0, 0)
            self.progress.setText('Relative progress · waiting for both trajectories')
            self._camera_state = None
            self._refit = True
            return
        self._selection(self.aa_view)
        maximum = max(len(v._reader.frames) for v in self.views) - 1
        self.slider.setRange(0, maximum)
        self.play.setEnabled(maximum > 0)
        self._seek(self.slider.value())

    def _selection(self, source):
        if self._syncing:
            return
        self._syncing = True
        try:
            target = self.cg_view if source is self.aa_view else self.aa_view
            target.scope.setCurrentIndex(source.scope.currentIndex())
            identity = source.species.currentData()
            if identity is not None:
                target.set_focus(identity, source.instance.value() - 1)
            target.dim.setChecked(source.dim.isChecked())
            target.follow.setChecked(source.follow.isChecked())
            self._refit = True
        finally:
            self._syncing = False

    def _request_refit(self, *_):
        self._refit = True

    def _display_options(self):
        for name in ('_pbc_enabled', '_show_box', '_single_show_box'):
            setattr(self.cg_view, name, getattr(self.aa_view, name))
        self.cg_view._reset_camera = self.aa_view._reset_camera
        self.cg_view._schedule()
        if self.aa_view.scope.currentIndex() == 0:
            self._refit = True

    def _frame_views(self):
        self._refit = True
        for view in self.views:
            view._reset_camera = True
            view._schedule()

    def _seek(self, position):
        maximum = self.slider.maximum()
        self.progress.setText('Relative progress · {:.0f}%'.format(100. * position / maximum if maximum else 0.))
        for view in self.views:
            count = len(view._reader.frames) if view._reader else 0
            view.set_frame(relative_frame(position, maximum, count))

    def _toggle_play(self):
        if self._timer.isActive():
            self._pause()
        elif self._ready() and self.slider.maximum() > 0:
            self._timer.start()
            self.play.setText('Pause both')

    def _advance(self):
        if not self._ready():
            self._pause()
        elif self._idle():
            # Wait for both readers/renderers instead of letting the faster one
            # run ahead. Shorter trajectories repeat their nearest frame.
            self.slider.setValue((self.slider.value() + 1) % (self.slider.maximum() + 1))

    def _camera_changed(self, state):
        self._camera_state = state
        self._refit = False
        for view in self.views:
            if view.renderer:
                view.renderer.set_linked_camera(state)

    def _scene_changed(self, source, reset):
        if self._refit and self._ready() and all(not v._busy and not v._dirty for v in self.views):
            state = self.aa_view.renderer.camera_state()
            state['distance'] = max(v.renderer.fit_distance for v in self.views)
            state['offset'] = [0., 0., 0.]
            self._camera_changed(state)
        elif self._camera_state is not None:
            # A late asynchronous scene update must not undo linked interaction.
            source.renderer.set_linked_camera(self._camera_state)

    def hideEvent(self, event):
        self._pause()
        super().hideEvent(event)

    def shutdown(self):
        self._pause()
        for view in self.views:
            view.shutdown()
