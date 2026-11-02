"""Portable archives and failure safety; no scientific engines are invoked."""
from copy import deepcopy
import errno
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from openmscg_gui.optimization_history import best_directory
from openmscg_gui.project import PROJECT_FILENAME, Project
from openmscg_gui.project_export import (
    ExportCancelled, _archive_name, plan_export, write_export,
)


class ProjectExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.project = Project.create(self.base / 'working project', 'Portable example')
        self.system = self.project.add_system('Molecule')
        self.root = self.project.system_root(self.system)
        self.external = self.base / 'external inputs'
        for field, filename, content in (
            ('trajectory', 'aa.lammpstrj', 'trajectory bytes; intentionally not parsed'),
            ('structure_file', 'aa.data', 'AA structure bytes'),
            ('aa_properties_file', 'aa.out', '# TimeStep temp\n0 298\n'),
        ):
            path = self.put(self.external / filename, content)
            setattr(self.system, field, str(path))
        self.system.custom_mapping_rules = [dict(bead_label='A', prototype_atoms=[0, 1])]
        self.system.stages.update(inspect='complete', cgmodel='complete', force='complete',
                                  postprocess='complete', lammps='complete', optimize='complete', validate='complete')
        self.opt = self.root / 'optimization'
        self.study = self.opt / 'studies' / 'study_example'
        self.best = self.study / 'best' / 'trial_0002'
        self.put(self.opt / 'active_study.json', dict(study='studies/study_example'))
        self.put(self.study / 'state.json', dict(best_number=2, best_score=.1,
            best_path='studies/study_example/best/trial_0002', fingerprint='unchanged', milestones=[0, 2]))
        for name, content in (('input.data', 'LAMMPS script'), ('structure.data', 'CG structure'),
                              ('Pair_A-A.table', 'exact best table bytes'),
                              ('cg.lammpstrj', 'exact best trajectory bytes'),
                              ('simulation_data.out', 'exact best properties bytes')):
            self.put(self.best / name, content)
        self.put(self.best / 'score.json', dict(score=.1, failed=False))
        self.put(self.best / 'correction.json', dict(trial=2))
        self.put(self.study / 'source' / 'input.data', 'source snapshot must remain exact')
        self.put(self.study / 'source' / 'structure.data', 'source CG structure')
        for number in range(3):
            self.put(self.study / 'history' / ('trial_{:04d}.json'.format(number)), dict(number=number))
            self.put(self.study / 'reservations' / ('trial_{:04d}.json'.format(number)), dict(number=number))
        other = self.study / 'runs' / 'trial_0000'
        self.put(other / 'cg.lammpstrj', 'other trial heavy trajectory')
        self.put(other / 'simulation_data.out', 'other trial property time series')
        self.put(other / 'Pair_A-A.table', 'other trial table')
        self.put(other / 'score.json', dict(score=.5))
        self.put(other / 'correction.json', dict(trial=0))
        archived = self.study / 'trial_records' / 'run_previous' / 'trial_0001'
        self.put(archived / 'trial_record.json', dict(trial=1, successful=False))
        self.put(self.root / 'output' / 'work' / 'types_cg.lammpstrj', 'mapped reference')
        self.put(self.root / 'output' / 'lammps' / 'input.data', 'base LAMMPS script')
        self.put(self.root / 'output' / 'lammps' / 'structure.data', 'base CG structure')
        self.put(self.root / 'output' / 'processed_tables' / '2_accepted' / 'Pair_A-A.table', 'prepared table')
        old_root = '/old/Linux/machine/project/'
        self.put(self.root / 'validation' / 'metrics.json', dict(summary=dict(rdf_distance=.02), sources={
            'mapped_aa': old_root + (self.root / 'output' / 'work' / 'types_cg.lammpstrj').relative_to(self.project.root).as_posix(),
            'cg': old_root + (self.best / 'cg.lammpstrj').relative_to(self.project.root).as_posix(),
            'mapped_aa_properties': self.system.aa_properties_file,
            'cg_result': 'per-system optimization'}))
        self.put(self.opt / '.study.lock', 'not bundled')
        self.put(self.root / '__pycache__' / 'cache.pyc', 'not bundled')
        self.put(self.project.root / 'unrelated-secret.txt', 'not application-managed')
        self.project.save()

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding='utf-8')
        return path

    def archive(self, preset='best'):
        path = self.base / (preset + '.zip')
        plan = plan_export(self.project, preset)
        write_export(plan, path)
        return path, plan

    def test_best_keeps_exact_winner_and_light_history_without_modifying_sources(self):
        before = {path: path.read_bytes() for path in self.base.rglob('*') if path.is_file()}
        state = deepcopy(self.project)
        destination, plan = self.archive()
        with zipfile.ZipFile(destination) as archive:
            names = set(archive.namelist())
            best = self.best.relative_to(self.project.root).as_posix()
            other = (self.study / 'runs' / 'trial_0000').relative_to(self.project.root).as_posix()
            self.assertEqual(archive.read(best + '/Pair_A-A.table'), b'exact best table bytes')
            self.assertIn(best + '/cg.lammpstrj', names)
            self.assertIn(other + '/score.json', names)
            self.assertNotIn(other + '/cg.lammpstrj', names)
            self.assertNotIn(other + '/simulation_data.out', names)
            self.assertNotIn(other + '/Pair_A-A.table', names)
            self.assertNotIn('unrelated-secret.txt', names)
            self.assertFalse(any('.study.lock' in name or '__pycache__' in name for name in names))
            snapshot = (self.study / 'source' / 'input.data').relative_to(self.project.root).as_posix()
            self.assertEqual(archive.read(snapshot), b'source snapshot must remain exact')
            manifest = json.loads(archive.read('export-manifest.json'))
            self.assertEqual(manifest['preset'], 'best')
            self.assertEqual(manifest['omitted_trial_files'], 3)
            for item in manifest['files']:
                data = archive.read(item['path'])
                self.assertEqual(len(data), item['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(), item['sha256'])
            self.assertIsNone(archive.testzip())
        self.assertEqual(self.project, state)
        self.assertEqual(plan.omitted_files, 3)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_extracted_project_can_move_and_be_saved_without_losing_relative_inputs(self):
        destination, _ = self.archive()
        extracted = self.base / 'different machine'
        with zipfile.ZipFile(destination) as archive:
            archive.extractall(extracted)
        restored = Project.load(extracted)
        item = restored.system(self.system.id)
        for field in ('trajectory', 'structure_file', 'aa_properties_file'):
            path = Path(getattr(item, field))
            self.assertTrue(path.is_file())
            self.assertIn(extracted, path.parents)
        self.assertEqual(best_directory(restored.system_root(item) / 'optimization'),
                         extracted / self.best.relative_to(self.project.root))
        self.assertEqual(restored.validation_trajectory(item).read_bytes(), b'exact best trajectory bytes')
        metrics = json.loads((restored.system_root(item) / 'validation' / 'metrics.json').read_text())
        self.assertEqual(metrics['summary'], dict(rdf_distance=.02))
        self.assertEqual(metrics['sources_relative_to'], 'project')
        for key in ('mapped_aa', 'cg', 'mapped_aa_properties'):
            self.assertTrue((extracted / metrics['sources'][key]).is_file())
        restored.save()
        raw = json.loads(restored.path.read_text())
        self.assertFalse(Path(raw['systems'][0]['trajectory']).is_absolute())
        moved = self.base / 'moved again'
        extracted.rename(moved)
        self.assertTrue(Path(Project.load(moved).systems[0].trajectory).is_file())

    def test_inputs_only_keeps_rules_and_resets_generated_stage_statuses(self):
        destination, _ = self.archive('inputs')
        with zipfile.ZipFile(destination) as archive:
            raw = json.loads(archive.read(PROJECT_FILENAME))
            self.assertFalse(any(name.startswith('systems/') or name.startswith('merge/') for name in archive.namelist()))
            self.assertEqual(raw['systems'][0]['custom_mapping_rules'], self.system.custom_mapping_rules)
            self.assertEqual(set(raw['systems'][0]['stages'].values()), {'pending'})
            self.assertEqual(raw['merge_status'], 'pending')

    def test_everything_keeps_other_trial_data(self):
        destination, _ = self.archive('all')
        with zipfile.ZipFile(destination) as archive:
            name = (self.study / 'runs' / 'trial_0000' / 'cg.lammpstrj').relative_to(self.project.root).as_posix()
            self.assertEqual(archive.read(name), b'other trial heavy trajectory')

    def test_input_shared_by_systems_is_stored_once(self):
        other = self.project.add_system('Second')
        other.trajectory = self.system.trajectory
        self.project.save()
        plan = plan_export(self.project, 'inputs')
        raw = json.loads(next(entry.data for entry in plan.entries if entry.name == PROJECT_FILENAME))
        self.assertEqual(raw['systems'][0]['trajectory'], raw['systems'][1]['trajectory'])
        self.assertEqual(sum(entry.source == Path(other.trajectory) for entry in plan.entries), 1)

    def test_missing_inputs_and_running_jobs_are_refused(self):
        self.system.aa_properties_file = str(self.base / 'missing.out')
        with self.assertRaisesRegex(FileNotFoundError, 'Relink'):
            plan_export(self.project)
        self.system.stages['optimize'] = 'running'
        with self.assertRaisesRegex(ValueError, 'running jobs'):
            plan_export(self.project)

    def test_invalid_best_is_not_exported_as_a_working_result(self):
        self.put(self.best / 'score.json', dict(score=1.e12, failed=True))
        with self.assertRaisesRegex(ValueError, 'scoring limits'):
            plan_export(self.project)
        self.assertTrue(plan_export(self.project, 'all').entries)

    def test_changed_source_aborts_and_cleans_temporary_output(self):
        plan = plan_export(self.project)
        self.put(Path(self.system.trajectory), 'changed after planning')
        destination = self.base / 'failed.zip'
        with self.assertRaisesRegex(ValueError, 'changed during export'):
            write_export(plan, destination)
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.base.glob('.openmscg-export-*')))

    def test_existing_destination_and_concurrent_collision_are_not_overwritten(self):
        plan = plan_export(self.project)
        destination = self.put(self.base / 'existing.zip', 'keep this')
        with self.assertRaises(FileExistsError):
            write_export(plan, destination)
        self.assertEqual(destination.read_text(), 'keep this')
        destination = self.base / 'collision.zip'
        def collision(percent, name):
            if not destination.exists():
                self.put(destination, 'another writer')
        with self.assertRaises(FileExistsError):
            write_export(plan, destination, progress=collision)
        self.assertEqual(destination.read_text(), 'another writer')
        self.assertFalse(list(self.base.glob('.openmscg-export-*')))

    def test_cancel_during_streaming_removes_partial_archive(self):
        plan = plan_export(self.project)
        destination = self.base / 'cancelled.zip'
        stop = [False]
        def progress(percent, name):
            stop[0] = True
        with self.assertRaises(ExportCancelled):
            write_export(plan, destination, cancelled=lambda: stop[0], progress=progress)
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.base.glob('.openmscg-export-*')))

    def test_filesystems_without_hard_links_use_exclusive_copy(self):
        destination = self.base / 'fallback.zip'
        with patch('openmscg_gui.project_export.os.link', side_effect=OSError(errno.EOPNOTSUPP, 'unsupported')):
            write_export(plan_export(self.project), destination)
        with zipfile.ZipFile(destination) as archive:
            self.assertIsNone(archive.testzip())
        self.assertFalse(list(self.base.glob('.openmscg-export-*')))

    def test_links_inside_managed_data_are_rejected(self):
        link = self.root / 'linked-folder'
        try:
            link.symlink_to(self.external, target_is_directory=True)
        except OSError:
            self.skipTest('Creating symbolic links is not permitted on this machine.')
        with self.assertRaisesRegex(ValueError, 'links'):
            plan_export(self.project, 'all')

    def test_archive_names_reject_unsafe_or_nonportable_paths(self):
        for name in ('../escape', '/absolute', 'C:/drive', 'folder/CON.txt', 'bad:name',
                     'folder/trailing.', 'folder\\backslash'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _archive_name(name)

    def test_legacy_best_directory_remains_loadable(self):
        (self.opt / 'active_study.json').unlink()
        legacy = self.opt / 'best'
        self.best.rename(legacy)
        destination, _ = self.archive()
        with zipfile.ZipFile(destination) as archive:
            name = (legacy / 'cg.lammpstrj').relative_to(self.project.root).as_posix()
            self.assertEqual(archive.read(name), b'exact best trajectory bytes')

    def test_merged_best_is_kept_but_not_other_merged_trajectories(self):
        merge = self.project.root / 'merge'
        self.project.merge_status = 'complete'
        self.project.merge_system_ids = [self.system.id]
        self.put(merge / 'best' / 'score.json', dict(score=.2, systems=[dict(score=.2, failed=False)]))
        self.put(merge / 'best' / 'systems' / self.system.name / 'cg.lammpstrj', 'merged winner')
        self.put(merge / 'runs' / 'trial_0000' / 'cg.lammpstrj', 'other merged trajectory')
        self.put(merge / 'runs' / 'trial_0000' / 'score.json', dict(score=.3))
        self.project.save()
        destination, _ = self.archive()
        with zipfile.ZipFile(destination) as archive:
            name = 'merge/best/systems/{}/cg.lammpstrj'.format(self.system.name)
            self.assertEqual(archive.read(name), b'merged winner')
            self.assertNotIn('merge/runs/trial_0000/cg.lammpstrj', archive.namelist())
            self.assertIn('merge/runs/trial_0000/score.json', archive.namelist())

    def test_failed_attempt_without_a_completed_best_can_still_be_shared(self):
        self.system.stages['optimize'] = 'failed'
        self.put(self.best / 'score.json', dict(score=1.e12, failed=True))
        self.project.save()
        plan = plan_export(self.project)
        name = (self.best / 'cg.lammpstrj').relative_to(self.project.root).as_posix()
        self.assertNotIn(name, [entry.name for entry in plan.entries])

    def test_omitted_historical_validation_source_is_not_replaced_with_the_winner(self):
        path = self.root / 'validation' / 'metrics.json'
        self.put(path, dict(summary=dict(rdf_distance=.9), sources=dict(
            cg=str(self.study / 'runs' / 'trial_0000' / 'cg.lammpstrj'))))
        plan = plan_export(self.project)
        value = json.loads(next(entry.data for entry in plan.entries if entry.source == path))
        self.assertEqual(value['sources']['cg'], '')
        self.assertEqual(value['summary']['rdf_distance'], .9)
        self.assertTrue(value['export_original_sources']['cg'])

    def test_linked_input_inside_managed_outputs_keeps_its_canonical_location(self):
        structure = self.put(self.root / 'output' / 'prepared' / 'aa_structure.data', 'inferred AA structure')
        self.system.structure_file = str(structure)
        self.project.save()
        plan = plan_export(self.project)
        name = structure.relative_to(self.project.root).as_posix()
        self.assertEqual(sum(entry.name == name for entry in plan.entries), 1)
        value = json.loads(next(entry.data for entry in plan.entries if entry.name == PROJECT_FILENAME))
        self.assertEqual(value['systems'][0]['structure_file'], name)

    def test_changing_the_saved_project_after_planning_is_detected(self):
        plan = plan_export(self.project)
        self.project.name = 'Changed in another window'
        self.project.save()
        with self.assertRaisesRegex(ValueError, 'changed during export'):
            write_export(plan, self.base / 'changed-project.zip')
        self.assertFalse((self.base / 'changed-project.zip').exists())


if __name__ == '__main__':
    unittest.main()
