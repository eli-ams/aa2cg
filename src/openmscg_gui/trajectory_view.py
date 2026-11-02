"""Lazy, asynchronous trajectory widget shared by Inspect, Map and Validate."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from threading import Event
from threading import RLock
from collections import OrderedDict

from PySide6 import QtCore, QtWidgets


_readers = OrderedDict()
_reader_lock = RLock()


class SharedTrajectory:
    def __init__(self, reader):
        self.reader = reader
        self.frames = reader.frames
        self.lock = RLock()

    def load_frame(self, index):
        with self.lock:
            return self.reader.load_frame(index)


def indexed_trajectory(path, cancelled):
    from .trajectory_gpu import LammpsDumpTrajectory
    key = file_stamp(path)
    # Only completed indices enter the cache. A cancelled source cannot poison it.
    with _reader_lock:
        if cancelled():
            raise InterruptedError('Trajectory indexing cancelled')
        if key not in _readers:
            _readers[key] = SharedTrajectory(LammpsDumpTrajectory(path).build_index(cancelled))
            while len(_readers) > 2:
                _readers.popitem(last=False)
        _readers.move_to_end(key)
        return _readers[key]


def file_stamp(path):
    path = Path(path)
    try:
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns
    except OSError:
        return str(path), None, None


class TrajectoryView(QtWidgets.QWidget):
    focusChanged = QtCore.Signal(str, int)
    frameChanged = QtCore.Signal(int)
    stateChanged = QtCore.Signal()
    sceneChanged = QtCore.Signal(bool)
    displayOptionsChanged = QtCore.Signal()
    frameViewRequested = QtCore.Signal()

    def __init__(self, parent=None, *, mapping=False, cg=False, compact_controls=True):
        super().__init__(parent)
        self.cg = cg
        self.comparison_mode = False
        self._key = None
        self._source = None
        self._generation = 0
        self._busy = False
        self._model = self._reader = self._frame = None
        self._pending_focus = None
        self._pending_frame = 0
        self._reset_camera = True
        self._dirty = False
        self._graphics_failure = None
        self._preview_error = False
        # Viewer-local preferences only: never written into a system or trajectory.
        self._pbc_enabled = True
        self._show_box = True
        self._single_show_box = False
        self._jobs = set()
        self._cancel = Event()
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._pool.setExpiryTimeout(1000)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.controls = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(self.controls)
        grid.setContentsMargins(0, 0, 0, 0)
        self.scope = QtWidgets.QComboBox()
        self.scope.addItems(['Whole system', 'Selected species', 'Single molecule'])
        self.scope.setToolTip('Hide everything except a species or one persistent molecule instance.')
        self.species = QtWidgets.QComboBox()
        self.species.setMinimumContentsLength(10)
        self.species.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.instance = QtWidgets.QSpinBox()
        self.instance.setPrefix('Molecule ')
        self.instance.setRange(1, 1)
        self.instance.setFixedWidth(125)
        choose = QtWidgets.QToolButton(text='Random')
        self.representation = QtWidgets.QComboBox()
        self.representation.addItem('All atoms', 'aa')
        if mapping:
            self.representation.addItem('AA + translucent beads', 'overlay')
            self.representation.addItem('Beads + unmapped atoms', 'cg')
            self.representation.setCurrentIndex(1)
        if cg:
            self.representation.setItemText(0, 'CG simulation')
        self.dim = QtWidgets.QCheckBox('Dim surroundings', self)
        self.follow = QtWidgets.QCheckBox('Follow molecule', self)
        self.follow.setChecked(True)
        self.dim.hide()
        self.follow.hide()
        grid.addWidget(self.scope, 0, 0)
        grid.addWidget(self.species, 0, 1)
        options_button = QtWidgets.QToolButton(text='View…')
        options_button.clicked.connect(lambda: self._show_view_menu(options_button))
        grid.addWidget(options_button, 0, 2)
        second_row = QtWidgets.QHBoxLayout()
        if mapping:
            second_row.addWidget(self.representation, 1)
        else:
            self.representation.hide()
        self.instance_controls = QtWidgets.QWidget()
        instance_controls_layout = QtWidgets.QHBoxLayout(self.instance_controls)
        instance_controls_layout.setContentsMargins(0, 0, 0, 0)
        instance_controls_layout.addWidget(self.instance)
        instance_controls_layout.addWidget(choose)
        second_row.addWidget(self.instance_controls)
        grid.addLayout(second_row, 1, 0, 1, 3)
        self.controls.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self.message = QtWidgets.QLabel('Load and inspect a trajectory to explore it in 3D.')
        self.message.setWordWrap(True)
        self.message.setObjectName('pageDescription')
        self.message.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.renderer = None
        try:
            from .trajectory_gpu import MolecularViewerWidget
            self.renderer = MolecularViewerWidget()
            self.renderer.initializationFailed.connect(self._graphics_error)
            self.renderer.setMinimumSize(200, 100)
            layout.addWidget(self.renderer, 1)
        except ImportError as exc:
            self.message.setText(f'3D preview requires the GUI dependencies (including moderngl): {exc}')
        layout.addWidget(self.message)
        self.playback_bar = QtWidgets.QWidget(self)
        row = QtWidgets.QHBoxLayout(self.playback_bar)
        row.setContentsMargins(0, 0, 0, 0)
        self.play = QtWidgets.QToolButton(text='Play')
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.frame_label = QtWidgets.QLabel('No frames')
        row.addWidget(self.play)
        row.addWidget(self.slider, 1)
        row.addWidget(self.frame_label)
        if compact_controls:
            self.controls.hide()
            self.view_button = QtWidgets.QToolButton(text='View ▾')
            self.view_button.setToolTip('Scope, species, molecule, representation and camera options')
            self.view_button.clicked.connect(lambda: self._show_view_menu(self.view_button))
            row.addWidget(self.view_button)
        layout.addWidget(self.playback_bar)
        if not compact_controls:
            layout.addWidget(self.controls)
        self.setToolTip('Drag: rotate · Right drag: pan · Wheel: zoom · Double-click: focus molecule')
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._advance)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(60)
        self._debounce.timeout.connect(self._request_frame)
        self.play.clicked.connect(self._toggle_play)
        self.slider.valueChanged.connect(self._schedule)
        self.slider.valueChanged.connect(self.frameChanged.emit)
        self.scope.currentIndexChanged.connect(self._selection_changed)
        self.species.currentIndexChanged.connect(self._species_changed)
        self.instance.valueChanged.connect(self._selection_changed)
        self.representation.currentIndexChanged.connect(self._selection_changed)
        self.dim.toggled.connect(self._selection_changed)
        self.follow.toggled.connect(self._selection_changed)
        choose.clicked.connect(self._random_molecule)
        if self.renderer:
            self.renderer.installEventFilter(self)
        self.controls.setEnabled(False)
        self.play.setEnabled(False)

    def _show_view_menu(self, anchor):
        menu = QtWidgets.QMenu(self)
        menu.setToolTipsVisible(True)
        ready = self._model is not None

        def choices(title, combo, enabled=True):
            submenu = menu.addMenu(title)
            submenu.setEnabled(ready and enabled)
            for index in range(combo.count()):
                action = submenu.addAction(combo.itemText(index))
                action.setCheckable(True)
                action.setChecked(index == combo.currentIndex())
                action.triggered.connect(lambda checked=False, i=index: combo.setCurrentIndex(i))

        choices('Show', self.scope)
        choices('Species', self.species, self.scope.currentIndex() != 0)
        molecule = menu.addMenu(f'Molecule {self.instance.value()} of {self.instance.maximum()}')
        molecule.setEnabled(ready)
        previous = molecule.addAction('Previous molecule')
        previous.setEnabled(self.instance.value() > 1)
        previous.triggered.connect(lambda: self._choose_molecule(self.instance.value() - 1))
        following = molecule.addAction('Next molecule')
        following.setEnabled(self.instance.value() < self.instance.maximum())
        following.triggered.connect(lambda: self._choose_molecule(self.instance.value() + 1))
        molecule.addAction('Random molecule', self._random_molecule)
        molecule.addAction('Go to molecule…', self._go_to_molecule)
        if self.representation.count() > 1:
            choices('Representation', self.representation)
        menu.addSeparator()
        for checkbox in (self.dim, self.follow):
            action = menu.addAction(checkbox.text())
            action.setCheckable(True)
            action.setChecked(checkbox.isChecked())
            action.setEnabled(ready and self.scope.currentIndex() != 0)
            action.toggled.connect(checkbox.setChecked)
        menu.addSeparator()
        source = self._frame if self._frame is not None else (self._model.scene if ready else None)
        has_cell = source is not None and source.cell_matrix is not None
        isolated = self.scope.currentIndex() == 2
        box = menu.addAction('Show simulation box')
        box.setCheckable(True)
        box.setChecked(self._display_options()['show_box'])
        box.setEnabled(ready and has_cell)
        box.setToolTip('Display the current frame’s cell edges; no simulation data is changed.'
                       if has_cell else 'No simulation cell is available for this view.')
        box.toggled.connect(self._set_box_display)
        pbc = menu.addAction('Make molecules whole (PBC)')
        pbc.setCheckable(True)
        pbc.setChecked(self._display_options()['pbc'])
        pbc.setEnabled(ready and has_cell and not isolated)
        pbc.setToolTip('Whole-molecule display is required while viewing one molecule.' if isolated else
                       'Display only: reconnect boundary-split molecules. Off shows coordinates as stored.'
                       if has_cell else 'No simulation cell is available for this view.')
        pbc.toggled.connect(self._set_pbc_display)
        frame = menu.addAction('Frame view', self._frame_view)
        frame.setEnabled(ready)
        menu.exec(anchor.mapToGlobal(QtCore.QPoint(0, anchor.height())))
        menu.deleteLater()

    def _display_options(self):
        isolated = self.scope.currentIndex() == 2
        return dict(pbc=isolated or self._pbc_enabled,
                    show_box=self._single_show_box if isolated else self._show_box)

    def _frame_view(self):
        self._selection_changed()
        self.frameViewRequested.emit()

    def _set_box_display(self, enabled):
        if self.scope.currentIndex() == 2:
            self._single_show_box = bool(enabled)
        else:
            self._show_box = bool(enabled)
        if self.scope.currentIndex() == 0:
            self._reset_camera = True
        self._schedule()
        self.displayOptionsChanged.emit()

    def _set_pbc_display(self, enabled):
        if self.scope.currentIndex() == 2:
            return  # Retain the full-system preference while isolation forces PBC on.
        self._pbc_enabled = bool(enabled)
        self._schedule()
        self.displayOptionsChanged.emit()

    def _choose_molecule(self, number):
        self.scope.setCurrentIndex(2)
        self.instance.setValue(number)

    def _go_to_molecule(self):
        number, accepted = QtWidgets.QInputDialog.getInt(
            self, 'Molecule focus', 'Molecule within the selected species:',
            self.instance.value(), 1, self.instance.maximum(),
        )
        if accepted:
            self._choose_molecule(number)

    def set_source(self, system, project_root, *, cg_structure=None, cg_trajectory=None):
        def resolve(value):
            path = Path(value) if value else Path(project_root) / '__missing__'
            return path if path.is_absolute() else Path(project_root) / path
        source = dict(structure=resolve(system.structure_file), trajectory=resolve(system.trajectory),
                      summaries=deepcopy(system.detected_species), mode=system.mapping_mode,
                      rules=deepcopy(system.custom_mapping_rules), cg_structure=cg_structure, cg_trajectory=cg_trajectory)
        stamps = [file_stamp(p) for p in (source['structure'], source['trajectory'], cg_structure, cg_trajectory) if p]
        key = json.dumps([stamps, source['summaries'], source['mode'], source['rules']], sort_keys=True)
        if key == self._key:
            return
        self._key, self._source = key, source
        self._preview_error = False
        self._cancel.set()
        self._cancel = Event()
        self._generation += 1
        self._model = self._reader = self._frame = None
        self._pending_frame = 0
        self._timer.stop()
        self.play.setText('Play')
        self.play.setEnabled(False)
        self.controls.setEnabled(False)
        self.slider.setRange(0, 0)
        self.frame_label.setText('No frames')
        self.message.show()
        self.stateChanged.emit()
        if self.renderer:
            from .trajectory_gpu import SceneData
            import numpy as np
            self.renderer.overlay_start = None
            self.renderer.set_scene(SceneData(np.empty((0, 7), dtype='f4'), np.empty((0, 14), dtype='f4'),
                                             0, np.zeros(3, dtype='f4'), 1., 'No structure'))
            self.message.setText('Preview will load when this view is opened.')
            self._ensure_loaded()

    def showEvent(self, event):
        super().showEvent(event)
        self._ensure_loaded()
        if self._model and self._dirty:
            self._request_frame()

    def _graphics_error(self, message):
        self._graphics_failure = message
        self.message.setText(message)
        self.message.setToolTip(message)
        self._timer.stop()
        self.play.setEnabled(False)
        self.message.show()
        self.stateChanged.emit()

    def hideEvent(self, event):
        self._timer.stop()
        self.play.setText('Play')
        super().hideEvent(event)

    def _start(self, kind, operation):
        from .trajectory_gpu import BackgroundWorker
        generation = self._generation
        def work():
            try:
                return generation, kind, operation(), None
            except Exception as exc:
                return generation, kind, None, str(exc)
        job = BackgroundWorker(work)
        self._jobs.add(job)
        self._busy = True
        job.signals.result.connect(self._result)
        job.signals.finished.connect(lambda: self._jobs.discard(job))
        self._pool.start(job)

    def _ensure_loaded(self):
        if not self.renderer or not self.isVisible() or not self._source or self._busy or self._model:
            return
        source = self._source.copy()
        if not source['structure'].is_file() or not source['summaries']:
            self.message.setText('Run Inspect to detect species and prepare a structure for 3D viewing.')
            return
        if self.cg and (not source['cg_structure'] or not source['cg_trajectory']):
            self.message.setText('No CG simulation is available. Run optional Refine, or simulate the generated LAMMPS setup externally (output/lammps/cg.lammpstrj). The smoke test creates no trajectory.')
            return
        self.message.setText('Loading structure and indexing trajectory…')
        cg = self.cg
        cancelled = self._cancel.is_set
        def load():
            from .trajectory_scene import load_model, load_cg_model
            model = load_model(source['structure'], source['summaries'], source['mode'], source['rules'])
            trajectory = source['trajectory']
            if cg:
                model = load_cg_model(source['cg_structure'], model)
                trajectory = Path(source['cg_trajectory'])
            reader = indexed_trajectory(trajectory, cancelled) if trajectory.is_file() else None
            return model, reader, str(trajectory)
        self._start('load', load)

    @QtCore.Slot(object)
    def _result(self, payload):
        generation, kind, value, error = payload
        self._busy = False
        if generation != self._generation:
            self._ensure_loaded()
            return
        if error:
            self._preview_error = True
            self._timer.stop()
            self.play.setText('Play')
            self.message.setText(f'3D preview unavailable: {error}')
            self.message.show()
            self.play.setEnabled(False)
            self.stateChanged.emit()
            return
        if kind == 'load':
            self._model, self._reader, source_path = value
            pending_focus = self._pending_focus
            was_blocked = self.blockSignals(True)
            self.species.blockSignals(True)
            self.species.clear()
            for s in self._model.species:
                self.species.addItem(f'{s.name} ({len(s.molecules)})', s.identity)
            self.species.blockSignals(False)
            self.controls.setEnabled(True)
            self._species_changed()
            if pending_focus:
                self.set_focus(*pending_focus)
            self.blockSignals(was_blocked)
            count = len(self._reader.frames) if self._reader else 0
            self.slider.setRange(0, max(0, count - 1))
            self.slider.setValue(min(self._pending_frame, self.slider.maximum()))
            self.play.setEnabled(count > 1)
            prefix = ('Independent CG simulation · relative progress, not matched physical time'
                      if self.cg else 'AA trajectory · geometric bead centres')
            if self._model.mapping_error:
                prefix += ' Mapping preview: ' + self._model.mapping_error
            if not count:
                prefix += ' Structure only; no trajectory frames available.'
            self.message.setText(prefix)
            if self._graphics_failure:
                self.message.setText(self._graphics_failure)
                self.play.setEnabled(False)
            self.message.setToolTip(source_path + '\nBead radii are visual guides, not physical interaction radii.')
            self.message.setVisible(not self.comparison_mode or not count or
                                    bool(self._graphics_failure) or bool(self._model.mapping_error))
            self._reset_camera = True
            self.stateChanged.emit()
            self._request_frame()
        elif kind == 'frame':
            index, frame, scene, overlay, reset = value
            self._frame = frame
            self.renderer.overlay_start = overlay
            self.renderer.set_scene(scene, reset_camera=reset)
            total = len(self._reader.frames) if self._reader else 0
            self.frame_label.setText(f'{index + 1}/{total} · step {frame.timestep}' if frame else 'Structure')
            if self._dirty:
                self._request_frame()
            self.sceneChanged.emit(reset)

    def _schedule(self, *_):
        self._dirty = True
        self._debounce.start()

    def _request_frame(self):
        if self._busy or self._model is None or not self.isVisible():
            return
        from .trajectory_scene import make_scene
        index, model, reader = self.slider.value(), self._model, self._reader
        options = dict(scope=self.scope.currentIndex(), species_index=max(0, self.species.currentIndex()),
                       molecule_index=self.instance.value() - 1, representation=self.representation.currentData(),
                       dim=self.dim.isChecked(), follow=self.follow.isChecked())
        options.update(self._display_options())
        reset = self._reset_camera
        self._reset_camera = self._dirty = False
        def render():
            frame = reader.load_frame(index) if reader and reader.frames else None
            scene, overlay = make_scene(model, frame, **options)
            return index, frame, scene, overlay, reset
        self._start('frame', render)

    def _species_changed(self, *_):
        if self._model and self.species.currentIndex() >= 0:
            species = self._model.species[self.species.currentIndex()]
            self.instance.blockSignals(True)
            self.instance.setRange(1, max(1, len(species.molecules)))
            self.instance.setValue(1)
            self.instance.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self, *_):
        self._reset_camera = True
        self.species.setEnabled(self.scope.currentIndex() != 0)
        self.instance.setEnabled(self.scope.currentIndex() == 2)
        self.instance_controls.setVisible(self.scope.currentIndex() == 2)
        self.dim.setEnabled(self.scope.currentIndex() != 0)
        if self._model and self.species.currentData() and self.scope.currentIndex():
            self._pending_focus = (self.species.currentData(), self.instance.value() - 1)
            species = self._model.species[self.species.currentIndex()]
            self.instance.setToolTip(f'Topology molecule ID: {species.molecule_ids[self.instance.value() - 1]}')
            self.focusChanged.emit(self.species.currentData(), self.instance.value() - 1)
        self._schedule()

    def set_focus(self, identity, molecule=0):
        self._pending_focus = (identity, molecule)
        index = self.species.findData(identity)
        if index < 0:
            return
        was_blocked = self.blockSignals(True)
        self.species.setCurrentIndex(index)
        self.instance.setValue(molecule + 1)
        self.blockSignals(was_blocked)
        self._reset_camera = True
        self._schedule()

    def set_frame(self, index):
        self._pending_frame = index
        if self._reader:
            self.blockSignals(True)
            self.slider.setValue(min(index, self.slider.maximum()))
            self.blockSignals(False)

    def _random_molecule(self):
        self.scope.setCurrentIndex(2)
        self.instance.setValue(random.randint(1, self.instance.maximum()))

    def _toggle_play(self):
        if self._timer.isActive():
            self._timer.stop()
            self.play.setText('Play')
        else:
            self._timer.start()
            self.play.setText('Pause')

    def _advance(self):
        if not self._busy:
            self.slider.setValue((self.slider.value() + 1) % (self.slider.maximum() + 1))

    def shutdown(self):
        self._cancel.set()
        self._generation += 1
        self._timer.stop()
        self._debounce.stop()

    def clear(self):
        self.shutdown()
        self._preview_error = False
        self._key = self._source = self._model = self._reader = self._frame = None
        self._pending_focus = None
        self._pending_frame = 0
        self.controls.setEnabled(False)
        self.play.setEnabled(False)
        self.play.setText('Play')
        self.slider.setRange(0, 0)
        self.frame_label.setText('No frames')
        if self.renderer:
            import numpy as np
            from .trajectory_gpu import SceneData
            self.renderer.overlay_start = None
            self.renderer.set_scene(SceneData(np.empty((0, 7), dtype='f4'), np.empty((0, 14), dtype='f4'),
                                             0, np.zeros(3, dtype='f4'), 1., 'No structure'))
            self.message.setText('Load and inspect a trajectory to explore it in 3D.')
        self.message.show()
        self.stateChanged.emit()

    def eventFilter(self, obj, event):
        if obj is self.renderer and event.type() == QtCore.QEvent.Type.MouseButtonDblClick and self._model:
            self._pick(event.position())
            return True
        return super().eventFilter(obj, event)

    def _pick(self, point):
        # Pick AA positions (also bead members in a mapped view) using the current camera.
        import numpy as np
        from .trajectory_scene import ordered_positions
        positions = ordered_positions(self._model, self._frame, pbc=self._display_options()['pbc'])
        if self.scope.currentIndex() and self.follow.isChecked():
            s = self._model.species[max(0, self.species.currentIndex())]
            members = np.concatenate(s.molecules) if self.scope.currentIndex() == 1 else s.molecules[self.instance.value() - 1]
            positions -= positions[members].mean(axis=0)
        view, projection = self.renderer._camera()
        clip = np.column_stack([positions, np.ones(len(positions))]) @ view.T @ projection.T
        valid = clip[:, 3] > 0
        ndc = clip[:, :3] / np.maximum(clip[:, 3:4], 1e-8)
        xy = (ndc[:, :2] * [0.5, -0.5] + 0.5) * [self.renderer.width(), self.renderer.height()]
        distances = np.linalg.norm(xy - [point.x(), point.y()], axis=1)
        distances[~valid] = np.inf
        if not len(distances) or distances.min() > 24:
            return
        selected = int(distances.argmin())
        for si, species in enumerate(self._model.species):
            for mi, molecule in enumerate(species.molecules):
                if selected in molecule:
                    self.scope.setCurrentIndex(2)
                    self.species.setCurrentIndex(si)
                    self.instance.setValue(mi + 1)
                    return
