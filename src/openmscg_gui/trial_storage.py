"""Per-system storage policy and conservative, completed-trial housekeeping."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import stat
import tempfile

from .optimization_search import valid_score
from .trial_diagnostics import outcome_reason


DEFAULT_TRIAL_STORAGE = {'mode': 'all', 'threshold_percent': 5.0}


def normalize_trial_storage(values=None):
    result = dict(DEFAULT_TRIAL_STORAGE, **(values or {}))
    if set(result) != set(DEFAULT_TRIAL_STORAGE) or result['mode'] not in ('all', 'milestones', 'best'):
        raise ValueError('Unknown trial storage policy.')
    try:
        threshold = float(result['threshold_percent'])
    except (TypeError, ValueError):
        raise ValueError('Milestone improvement must be between 0 and 100%.')
    if not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError('Milestone improvement must be between 0 and 100%.')
    result['threshold_percent'] = threshold
    return result


def _linked(path):
    # Windows junctions/reparse points need the same protection as symlinks.
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) &
                                    getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024))


class TrialStorage:
    """Optuna callback; never discovers/deletes trials from previous runs.

    The optimizer is sequential. A callback runs only after its objective exits,
    including property and structural scoring, and before the next trial starts.
    """
    def __init__(self, work_dir, policy=None, protected_paths=()):
        self.work_dir = Path(work_dir).resolve()
        self.policy = normalize_trial_storage(policy)
        self.protected_paths = tuple(Path(p).resolve() for p in protected_paths)
        self.best_number = None
        self.best_score = None
        self.milestones = set()
        self.anchor_score = None
        self.finished = set()
        self.cleaned = set()
        self.record_dir = None
        self.archives = {}
        self.exported_best = None

    def _directory(self, number):
        path = self.work_dir / 'trial_{:04d}'.format(number)
        if (number < 0 or _linked(path) or path.resolve().parent != self.work_dir or
                any(p == path.resolve() or p in path.resolve().parents or
                    path.resolve() in p.parents for p in self.protected_paths)):
            raise ValueError('Refusing trial housekeeping outside an isolated trial directory.')
        return path

    def __call__(self, study, trial):
        path = self._directory(trial.number)
        score = None
        score_data = {}
        try:
            score_data = json.loads((path / 'score.json').read_text(encoding='utf-8'))
            if trial.state.name == 'COMPLETE':
                candidate = valid_score(score_data)
                if candidate is not None and trial.value == candidate:
                    score = candidate
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass  # Missing/incomplete scoring is not a successful candidate.

        record = dict(trial=trial.number, state=trial.state.name, value=trial.value,
                      parameters=dict(trial.params), user_attributes=dict(trial.user_attrs),
                      started=str(trial.datetime_start), finished=str(trial.datetime_complete),
                      successful=score is not None, storage_policy=self.policy)
        record['outcome_reason'] = outcome_reason(score_data)
        record_path = path / 'trial_record.json'
        if record_path.is_symlink() or (record_path.exists() and _linked(record_path)):
            raise ValueError('Refusing to overwrite a linked trial record.')
        # Persist the audit record before permitting any cleanup.
        record_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        self.finished.add(trial.number)

        if score is not None and (self.best_score is None or score < self.best_score):
            self.best_number, self.best_score = trial.number, score
            if (self.anchor_score is None or
                    self.anchor_score - score >= abs(self.anchor_score) *
                    self.policy['threshold_percent'] / 100.):
                self.milestones.add(trial.number)
                self.anchor_score = score

        if self.policy['mode'] == 'all':
            return
        keep = {self.best_number} if self.best_number is not None else set()
        if self.policy['mode'] == 'milestones':
            keep.update(self.milestones)
        for number in sorted(self.finished - keep - self.cleaned):
            try:
                removed = self._prune(number)
                self.cleaned.add(number)
                print('Trial {}: records/logs archived to {}; removed {} files ({:.1f} MiB). '
                      'Removed data cannot be recovered.'.format(number, self.archives[number], *removed))
            except (OSError, ValueError) as exc:
                # A locked file or unsafe path must not destroy the optimization result.
                print('Trial {} cleanup skipped/incomplete: {}'.format(number, exc))

    @staticmethod
    def _tree(root):
        """Inspect the whole tree before deleting anything; reject all links."""
        files, directories = [], []

        def visit(directory):
            if _linked(directory):
                raise ValueError('Refusing housekeeping through a link/junction.')
            for entry in directory.iterdir():
                if _linked(entry) or root not in entry.resolve().parents:
                    raise ValueError('Unsafe link or path in trial files.')
                if entry.is_dir():
                    visit(entry)
                    directories.append(entry)
                elif entry.is_file():
                    files.append(entry)
                else:
                    raise ValueError('Unsupported trial file: {}'.format(entry))

        visit(root)
        return files, directories

    def _archive(self, number):
        base = self.work_dir.parent / 'trial_records'
        if base.is_symlink() or (base.exists() and _linked(base)):
            raise ValueError('Refusing a linked trial-record directory.')
        if self.record_dir is None:
            if any(p == base or p in base.parents or base in p.parents
                   for p in self.protected_paths):
                raise ValueError('Trial records overlap a protected directory.')
            base.mkdir(exist_ok=True)
            # A fresh namespace avoids overwriting diagnostics from previous studies.
            self.record_dir = Path(tempfile.mkdtemp(prefix='run_', dir=str(base)))
        if _linked(self.record_dir):
            raise ValueError('Refusing a linked trial-record directory.')
        directory = self.record_dir / 'trial_{:04d}'.format(number)
        if directory.is_symlink() or (directory.exists() and _linked(directory)):
            raise ValueError('Refusing a linked trial archive.')
        directory.mkdir(exist_ok=True)
        self.archives[number] = directory
        return directory

    def release_exported_best(self, destination):
        """Remove the redundant run only after the ordinary best export succeeds."""
        if self.policy['mode'] == 'all':
            return
        root = self._directory(self.best_number)
        destination = Path(destination)
        if _linked(destination):
            raise ValueError('Refusing a linked best export.')
        destination = destination.resolve()
        if root == destination or root in destination.parents or destination in root.parents:
            raise ValueError('Best export overlaps the original trial.')
        source_files, _ = self._tree(root)
        target_files, _ = self._tree(destination)
        source_sizes = {str(p.relative_to(root)): p.stat().st_size for p in source_files}
        target_sizes = {str(p.relative_to(destination)): p.stat().st_size for p in target_files}
        record = 'trial_record.json'
        if (source_sizes != target_sizes or
                (root / record).read_bytes() != (destination / record).read_bytes()):
            raise ValueError('Best export is incomplete; keeping the original trial.')
        self.exported_best = self.best_number
        count, size = self._prune(self.best_number)
        self.cleaned.add(self.best_number)
        print('Best exported to {}; redundant trial removed ({} files, {:.1f} MiB).'
              .format(destination, count, size))

    def _prune(self, number):
        root = self._directory(number)
        exported = number == self.best_number == self.exported_best
        if (number not in self.finished or self.policy['mode'] == 'all' or
                (not exported and (number == self.best_number or
                 (self.policy['mode'] == 'milestones' and number in self.milestones)))):
            raise ValueError('Trial is protected or has not finished with an audit record.')
        files, directories = self._tree(root)
        record = root / 'trial_record.json'
        previous_archive = self.archives.get(number)
        if (not record.is_file() and (previous_archive is None or
                not (previous_archive / record.name).is_file())):
            raise ValueError('Trial has no durable audit record.')
        archive = self._archive(number)
        self._tree(archive)  # Refuse links left in any partially written archive.
        names = {'input.data', 'correction.json', 'score.json', 'trial_record.json',
                 'bead_signature.json'}
        count, size = len(files), sum(p.stat().st_size for p in files)
        # Finish every diagnostic copy BEFORE deleting a single original file.
        for entry in files:
            if (entry.name in names or entry.suffix.lower() == '.log' or
                    entry.name.lower().startswith('log.')):
                target = archive / entry.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or (target.exists() and _linked(target)):
                    raise ValueError('Unsafe file in trial archive.')
                shutil.copy2(str(entry), str(target))
        (archive / 'storage.json').write_text(json.dumps(dict(full_data_retained=False,
            removed_files=count, removed_bytes=size, policy=self.policy,
            exported_best=exported,
            reason='Full result exported to best.' if exported else
                   'Not a retained, valid milestone or current best.'), indent=2) + '\n', encoding='utf-8')
        # Recheck before destructive operations; only this callback's finished trials qualify.
        self._directory(number)
        # Keep the original audit record until all other files have been removed.
        for entry in sorted(files, key=lambda p: p == record):
            if _linked(entry) or root not in entry.resolve().parents:
                raise ValueError('Trial path changed during cleanup.')
            entry.unlink()
        for directory in directories:
            directory.rmdir()
        root.rmdir()
        return count, size / (1024. * 1024.)
