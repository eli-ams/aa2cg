from __future__ import annotations

import os
from pathlib import Path
import sys

from PySide6 import QtCore


class ProcessRunner(QtCore.QObject):
    outputReceived = QtCore.Signal(str)
    started = QtCore.Signal(str, object)
    finished = QtCore.Signal(str, object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self._stage = ""
        self._system_id = None

    @property
    def running(self) -> bool:
        return self.process.state() != QtCore.QProcess.ProcessState.NotRunning

    def start(self, project_path: Path, stage: str, system_id: str | None = None, continuation=None) -> None:
        if self.running:
            raise RuntimeError("Another workflow stage is already running.")
        self._stage = stage
        self._system_id = system_id

        environment = QtCore.QProcessEnvironment.systemEnvironment()
        src_root = str(Path(__file__).resolve().parents[1])
        existing = environment.value("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        environment.insert("PYTHONPATH", os.pathsep.join(part for part in [src_root, existing] if part))
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(project_path.parent))

        arguments = [
            "-u",
            "-m",
            "openmscg_gui.worker",
            "--project",
            str(project_path),
            "--stage",
            stage,
        ]
        if system_id:
            arguments.extend(["--system", system_id])
        if continuation is not None:
            arguments.extend(['--continue-study', '--add-explore', str(continuation[0]),
                              '--add-refine', str(continuation[1])])
        self.process.start(sys.executable, arguments)
        if not self.process.waitForStarted(3000):
            raise RuntimeError(self.process.errorString())
        self.started.emit(stage, system_id)

    def cancel(self) -> None:
        if not self.running:
            return
        self.outputReceived.emit("\nCancellation requested…\n")
        self.process.terminate()
        QtCore.QTimer.singleShot(5000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.running:
            self.process.kill()

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.outputReceived.emit(data)

    def _finished(self, exit_code: int, _exit_status) -> None:
        self._read_output()
        stage, system_id = self._stage, self._system_id
        self._stage = ""
        self._system_id = None
        self.finished.emit(stage, system_id, exit_code == 0)

