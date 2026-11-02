"""Portable project ZIPs. Read sources only; stream large files, never stage copies.

Only linked inputs and application-managed system/merge folders are included.
Scientific files and optimization fingerprints are not rewritten. The exported
project and validation navigation metadata are the only relocated JSON files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

from . import __version__
from .optimization_history import best_directory, read_json
from .optimization_search import valid_score
from .project import PROJECT_FILENAME, Project


PRESETS = (
    ('best', 'Best result', 'Inputs, settings and outputs; only the current best valid trial keeps its full data.'),
    ('inputs', 'Inputs + settings', 'Linked inputs and saved mapping/settings. Generated results are omitted; stages reset to pending.'),
    ('all', 'Everything', 'Linked inputs and all managed outputs, including every retained trial.'),
)
INPUT_FIELDS = ('trajectory', 'structure_file', 'aa_properties_file')
SKIP_DIRECTORIES = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv'}
LIGHT_TRIAL_FILES = {'score.json', 'correction.json', 'trial_record.json', 'storage.json',
                     'bead_signature.json', 'input.data'}
CHUNK_SIZE = 1024 * 1024


class ExportCancelled(Exception):
    pass


def _check_cancel(cancelled):
    if cancelled and cancelled():
        raise ExportCancelled('Export cancelled.')


def _signature(path):
    value = path.stat()
    return (value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino)


def _ordinary(path):
    # resolve() also catches Windows junctions, including on Python versions
    # without is_junction(). Cloud placeholders are not mistaken for links.
    if path.is_symlink() or path.absolute() != path.resolve():
        raise ValueError('Export does not follow links or junctions: {}'.format(path))
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Export requires an ordinary file: {}'.format(path))


def _archive_name(name):
    parts = PurePosixPath(name).parts
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {
        '{}{}'.format(prefix, number) for prefix in ('COM', 'LPT') for number in range(1, 10)}
    if (not parts or PurePosixPath(name).is_absolute() or '\\' in name or
            any(p in ('.', '..') or p.rstrip(' .') != p or
                p.split('.')[0].upper() in reserved or
                any(ord(c) < 32 or c in '<>:"|?*' for c in p) for p in parts)):
        raise ValueError('File name is not portable between Windows and Linux: {}'.format(name))
    return '/'.join(parts)


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


@dataclass(frozen=True)
class ExportEntry:
    name: str
    source: Path | None
    signature: tuple | None
    data: bytes | None = None

    @property
    def size(self):
        return len(self.data) if self.data is not None else self.signature[0]


@dataclass(frozen=True)
class ExportPlan:
    preset: str
    entries: tuple
    omitted_files: int
    versions: dict

    @property
    def total_bytes(self):
        return sum(entry.size for entry in self.entries)


def _walk_files(root, cancelled):
    if root.is_symlink() or root.absolute() != root.resolve():
        raise ValueError('Export does not follow links or junctions: {}'.format(root))
    if not root.exists():
        return
    stack = [root]
    while stack:
        _check_cancel(cancelled)
        directory = stack.pop()
        if directory.is_symlink() or directory.absolute() != directory.resolve():
            raise ValueError('Export does not follow links or junctions: {}'.format(directory))
        with os.scandir(str(directory)) as iterator:
            for item in iterator:
                _check_cancel(cancelled)
                path = Path(item.path)
                if item.is_symlink():
                    raise ValueError('Export does not follow links: {}'.format(path))
                if item.is_dir(follow_symlinks=False):
                    if item.name not in SKIP_DIRECTORIES:
                        stack.append(path)
                elif item.is_file(follow_symlinks=False):
                    if (item.name != '.study.lock' and not item.name.lower().endswith(('.tmp', '.zip', '.pyc'))
                            and '.tmp.' not in item.name):
                        yield path
                else:
                    raise ValueError('Export found a non-ordinary file: {}'.format(path))


def _versions():
    # Preserve the legacy key for consumers of existing export manifests.
    result = {'aa2cg': __version__, 'openmscg-gui': __version__,
              'python': sys.version, 'platform': sys.platform}
    try:
        try:
            from importlib import metadata
        except ImportError:
            import importlib_metadata as metadata
        for name in ('numpy', 'scipy', 'matplotlib', 'rdkit', 'optuna', 'PySide6', 'openmscg'):
            try:
                result[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                result[name] = 'package metadata unavailable'
    except ImportError:
        result['dependency_versions'] = 'package metadata unavailable'
    result['lammps'] = 'See archived LAMMPS logs; no executable was invoked during export.'
    return result


def _best_valid(root, required=False, merged=False):
    directory = root / 'best' if merged else best_directory(root)
    if not directory.is_dir():
        if required:
            raise ValueError('The completed best result is missing: {}'.format(directory))
        return None
    score_file = directory / 'score.json'
    if not score_file.is_file():
        if not required:
            return None
        raise ValueError('Cannot verify the best result; score.json is missing: {}'.format(directory))
    data = read_json(score_file)
    valid = valid_score(data) is not None
    if merged:
        valid = valid and bool(data.get('systems')) and all(
            valid_score(item) is not None for item in data['systems'])
    if not valid:
        if not required:
            return None
        raise ValueError('The saved best result did not pass scoring limits: {}'.format(directory))
    if not merged and not all((directory / name).is_file() for name in ('input.data', 'structure.data')):
        raise ValueError('The best result is missing its LAMMPS setup: {}'.format(directory))
    return directory


def _keep_best_file(path, root, best):
    if best is not None and (path == best or best in path.parents):
        return True
    parts = path.relative_to(root).parts
    # Source snapshots, immutable history and reservations remain intact, so
    # continuation retains its existing safety checks. Heavy trial data does not.
    if (parts[0] == 'source' or
            (len(parts) > 3 and parts[0] == 'studies' and parts[2] == 'source')):
        return True
    if any(part in ('runs', 'best', 'trial_records') for part in parts[:-1]):
        return path.name in LIGHT_TRIAL_FILES and path.stat().st_size <= CHUNK_SIZE
    return True


def _relocated_sources(value, project, entries):
    """Relocate navigation links only, never metrics or optimization settings."""
    by_source = {entry.source: entry.name for entry in entries if entry.source is not None}
    names = {entry.name for entry in entries}
    markers = ['systems/' + system.directory_name + '/' for system in project.systems] + ['merge/']
    original = dict(value.get('sources', {}))
    moved = {}
    for key, text in original.items():
        if key == 'cg_result' or not text:
            moved[key] = text
            continue
        candidate = Path(str(text))
        if not candidate.is_absolute():
            candidate = project.root / candidate
        target = by_source.get(candidate.resolve())
        if target is None:
            text_path = str(text).replace('\\', '/')
            for marker in markers:
                index = text_path.rfind('/' + marker)
                name = text_path[index + 1:] if index >= 0 else text_path
                if name in names:
                    target = name
                    break
        # Omitted historical trajectories must not be relabeled as the best.
        moved[key] = target or ''
    value['export_original_sources'] = value.get('export_original_sources', original)
    value['sources'] = moved
    value['sources_relative_to'] = 'project'
    return value


def plan_export(project: Project, preset='best', cancelled=None):
    _check_cancel(cancelled)
    if preset not in {item[0] for item in PRESETS}:
        raise ValueError('Unknown export preset.')
    if project.path is None:
        raise ValueError('Save the project before exporting.')
    if not project.path.is_file():
        raise FileNotFoundError('Save the project before exporting.')
    project_signature = _signature(project.path)
    if project.merge_status == 'running' or any(
            'running' in system.stages.values() for system in project.systems):
        raise ValueError('Wait for or cancel running jobs before exporting.')
    root = project.root.resolve()
    entries, names, sources = [], set(), {}
    omitted = 0

    def add(path, name):
        path = Path(os.path.abspath(str(path)))
        _ordinary(path)
        name = _archive_name(name)
        if name.casefold() in names:
            raise ValueError('Two export files have the same portable name: {}'.format(name))
        entry = ExportEntry(name, path, _signature(path))
        entries.append(entry)
        names.add(name.casefold())
        sources[path.resolve()] = name
        return name

    payload = asdict(project)
    payload.pop('path', None)
    system_dirs = set()
    for system, saved in zip(project.systems, payload['systems']):
        _check_cancel(cancelled)
        if system.directory_name in system_dirs:
            raise ValueError('Systems share an output folder; resolve duplicate system IDs first.')
        system_dirs.add(system.directory_name)
        for key in INPUT_FIELDS:
            text = getattr(system, key)
            if not text:
                continue
            path = Path(text).expanduser()
            path = path if path.is_absolute() else root / path
            path = Path(os.path.abspath(str(path)))
            if not path.is_file():
                raise FileNotFoundError('Relink the missing {} for {} before export: {}'.format(key, system.name, path))
            if path.resolve() not in sources:
                suffix = path.suffix or ('.lammpstrj' if key == 'trajectory' else '.data' if key == 'structure_file' else '.out')
                name = 'inputs/{}/{}{}'.format(system.directory_name, key, suffix)
                if preset != 'inputs':
                    try:
                        relative = path.relative_to(root)
                        if relative.parts[0] in ('systems', 'merge'):
                            # A generated file can also be an input. Keep its
                            # canonical location so other artifacts still find it.
                            name = relative.as_posix()
                    except ValueError:
                        pass
                add(path, name)
            saved[key] = sources[path.resolve()]
        if preset == 'inputs':
            saved['stages'] = {key: 'pending' for key in system.stages}
            saved['invalidated_stages'] = list(system.stages)
            continue
        system_root = root / 'systems' / system.directory_name
        if root not in system_root.resolve().parents:
            raise ValueError('System output folder escapes the project.')
        optimization = system_root / 'optimization'
        best = _best_valid(optimization, system.stages.get('optimize') == 'complete') if preset == 'best' else None
        for path in _walk_files(system_root, cancelled):
            if path.resolve() in sources:
                continue
            if (preset == 'best' and optimization in path.parents and
                    not _keep_best_file(path, optimization, best)):
                omitted += 1
                continue
            add(path, path.relative_to(root).as_posix())
    if preset == 'inputs':
        payload['merge_status'] = 'pending'
        payload['merge_invalidated'] = True
    else:
        merge = root / 'merge'
        best = _best_valid(merge, project.merge_status == 'complete', merged=True) if (
            preset == 'best' and (merge / 'best').exists()) else None
        for path in _walk_files(merge, cancelled):
            if path.resolve() in sources:
                continue
            if preset == 'best' and not _keep_best_file(path, merge, best):
                omitted += 1
                continue
            add(path, path.relative_to(root).as_posix())
    # Build relative validation links only after the complete file set is known.
    for index, entry in enumerate(entries):
        if entry.name.endswith('/validation/metrics.json'):
            value = _relocated_sources(read_json(entry.source), project, entries)
            entries[index] = ExportEntry(entry.name, entry.source, entry.signature, _json_bytes(value))
    entries.append(ExportEntry(PROJECT_FILENAME, project.path.resolve(), project_signature, _json_bytes(payload)))
    instructions = (
        '# Portable OpenMSCG project\n\n'
        'Extract this ZIP into a separate folder, then use File > Open project to open '
        'openmscg-project.json. Opening it does not run simulations.\n\n'
        'Preset: {}. Only referenced inputs and application-managed system/merge data are included. '
        'Unrelated files, caches, archive files and locks are not bundled.\n\n'
        'Best result keeps full data only for the current best valid trial per system and '
        'the merged best, plus lightweight trial records. Omitted trials cannot be reopened in full. '
        'Existing continuation checks still apply; no guarantee of identical resumed searches.\n\n'
        'Check the LAMMPS launch command on this machine before running jobs. Scientific source '
        'snapshots, logs and provenance may contain original paths; this is not a privacy scrubber.\n\n'
        'export-manifest.json records versions and SHA-256 checksums of the archived files. '
        'No force matching, simulation or software executable was run during export.\n'
    ).format(preset)
    entries.append(ExportEntry('EXPORT-README.md', None, None, instructions.encode('utf-8')))
    return ExportPlan(preset, tuple(sorted(entries, key=lambda entry: entry.name)), omitted, _versions())


def _verify(entry):
    if entry.source is not None:
        _ordinary(entry.source)
        if _signature(entry.source) != entry.signature:
            raise ValueError('Project file changed during export; retry when idle: {}'.format(entry.source))


def write_export(plan: ExportPlan, destination, cancelled=None, progress=None):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Choose a new archive name; existing files are never overwritten.')
    if not destination.parent.is_dir():
        raise FileNotFoundError('The archive destination folder does not exist.')
    if any(entry.source == destination for entry in plan.entries):
        raise ValueError('The archive destination cannot replace a project input.')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.openmscg-export-', suffix='.tmp',
                                         dir=str(destination.parent), delete=False) as stream:
            temporary = Path(stream.name)
        checksums, done, last_progress = [], 0, -1
        total = max(plan.total_bytes, 1)
        with zipfile.ZipFile(str(temporary), 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as archive:
            for entry in plan.entries:
                _check_cancel(cancelled)
                _verify(entry)
                digest = hashlib.sha256()
                with archive.open(entry.name, 'w', force_zip64=True) as target:
                    if entry.data is not None:
                        target.write(entry.data)
                        digest.update(entry.data)
                        done += len(entry.data)
                    else:
                        with entry.source.open('rb') as source:
                            while True:
                                _check_cancel(cancelled)
                                block = source.read(CHUNK_SIZE)
                                if not block:
                                    break
                                target.write(block)
                                digest.update(block)
                                done += len(block)
                                percent = min(99, int(100 * done / total))
                                if progress and percent != last_progress:
                                    progress(percent, entry.name)
                                    last_progress = percent
                _verify(entry)
                checksums.append(dict(path=entry.name, bytes=entry.size, sha256=digest.hexdigest(),
                                      relocated_metadata=entry.source is not None and entry.data is not None))
            manifest = dict(format_version=1, preset=plan.preset,
                            created_utc=datetime.now(timezone.utc).isoformat(),
                            versions=plan.versions, omitted_trial_files=plan.omitted_files, files=checksums)
            archive.writestr('export-manifest.json', _json_bytes(manifest))
        for entry in plan.entries:
            _check_cancel(cancelled)
            _verify(entry)
        _check_cancel(cancelled)
        with temporary.open('r+b') as stream:
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses collisions. On filesystems
        # without hard links, copy to an exclusively created destination instead.
        try:
            os.link(str(temporary), str(destination))
        except FileExistsError:
            raise
        except OSError:
            created = False
            try:
                with destination.open('xb') as target:
                    created = True
                    with temporary.open('rb') as source:
                        while True:
                            _check_cancel(cancelled)
                            block = source.read(CHUNK_SIZE)
                            if not block:
                                break
                            target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
            except BaseException:
                if created:
                    destination.unlink()
                raise
        if progress:
            progress(100, destination.name)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
