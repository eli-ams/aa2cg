from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from openmscg_gui.trajectory_gpu import LammpsDumpTrajectory, DumpFrameData, box_line_vertices
from openmscg_gui.trajectory_scene import load_model, make_scene, ordered_positions, load_cg_model, model_bead_colour


WATER = '''Two water molecules

6 atoms
4 bonds
2 atom types
1 bond types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 15.999 # O
2 1.008 # H

Atoms # full

1 1 1 0 9.8 2 2
2 1 2 0 0.75 2 2
3 1 2 0 9.55 2.92 2
4 2 2 0 5.95 5 5
5 2 1 0 5 5 5
6 2 2 0 4.75 5.92 5

Bonds

1 1 1 2
2 1 1 3
3 1 5 4
4 1 5 6
'''


class TrajectorySceneTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'water.data'
        self.path.write_text(WATER)
        self.rules = [dict(species_id='species-1', species_smiles='O', bead_label='W',
                           matches=[dict(atoms=[0, 1, 2])])]
        self.model = load_model(self.path, [], 'manual_custom', self.rules)

    def test_repeated_molecules_align_atoms_despite_different_element_order(self):
        self.assertEqual(len(self.model.species), 1)
        self.assertEqual(len(self.model.species[0].molecules), 2)
        self.assertEqual({frozenset(b) for b in self.model.beads},
                         {frozenset([0, 1, 2]), frozenset([3, 4, 5])})

    def test_boundary_molecule_is_whole_before_bead_centre(self):
        positions = ordered_positions(self.model, None)
        self.assertAlmostEqual(np.linalg.norm(positions[0] - positions[1]), .95, places=5)
        scene, overlay = make_scene(self.model, representation='overlay', scope=2)
        self.assertEqual(overlay, 3)
        self.assertEqual(len(scene.atom_data), 4)
        np.testing.assert_allclose(scene.atom_data[3, :3], scene.atom_data[:3, :3].mean(axis=0), atol=1e-6)

    def test_beads_replace_members_and_partial_mapping_retains_atoms(self):
        scene, _ = make_scene(self.model, representation='cg')
        self.assertEqual(len(scene.atom_data), 2)
        partial = [dict(species_id='species-1', species_smiles='O', bead_label='O', matches=[dict(atoms=[0])])]
        model = load_model(self.path, [], 'manual_custom', partial)
        scene, _ = make_scene(model, representation='cg')
        self.assertEqual(len(scene.atom_data), 6)  # Four H atoms and two O beads.
        self.assertEqual(len(scene.bond_data), 4)

    def test_frame_reordering_preserves_identity_and_does_not_modify_source(self):
        positions = ordered_positions(self.model, None)
        order = np.array([5, 2, 0, 4, 1, 3])
        frame = DumpFrameData(0, 10, self.model.scene.atom_ids[order], positions[order] + 3,
                              np.eye(3) * 10, np.ones(3, dtype=bool), False)
        original = frame.positions.copy()
        first, _ = make_scene(self.model, None, scope=2, molecule_index=1)
        moved, _ = make_scene(self.model, frame, scope=2, molecule_index=1)
        np.testing.assert_allclose(first.atom_data, moved.atom_data, atol=1e-6)
        np.testing.assert_array_equal(frame.positions, original)

    def test_duplicate_frame_ids_rejected(self):
        frame = DumpFrameData(0, 0, np.ones(6, dtype=int), np.zeros((6, 3)),
                              np.eye(3), np.ones(3, dtype=bool), False)
        with self.assertRaisesRegex(ValueError, 'atom IDs'):
            ordered_positions(self.model, frame)

    def test_pbc_display_toggle_and_isolation_do_not_modify_model(self):
        original = self.model.scene.atom_data.copy()
        wrapped, _ = make_scene(self.model, pbc=False)
        whole, _ = make_scene(self.model, pbc=True)
        self.assertGreater(np.linalg.norm(wrapped.atom_data[0, :3] - wrapped.atom_data[1, :3]), 9.)
        self.assertAlmostEqual(np.linalg.norm(whole.atom_data[0, :3] - whole.atom_data[1, :3]), .95, places=5)
        self.assertLess(len(wrapped.bond_data), len(whole.bond_data))
        isolated, _ = make_scene(self.model, scope=2, pbc=False)
        self.assertAlmostEqual(np.linalg.norm(isolated.atom_data[0, :3] - isolated.atom_data[1, :3]), .95, places=5)
        np.testing.assert_allclose(isolated.atom_data[:, :3].mean(axis=0), 0, atol=1e-6)
        np.testing.assert_array_equal(self.model.scene.atom_data, original)
        self.assertEqual(self.path.read_text(), WATER)

    def test_wrapped_beads_keep_whole_geometry(self):
        whole, _ = make_scene(self.model, representation='cg', pbc=True)
        wrapped, _ = make_scene(self.model, representation='cg', pbc=False)
        np.testing.assert_allclose(wrapped.atom_data[:, 3:], whole.atom_data[:, 3:])
        cell = self.model.scene.cell_matrix
        difference = (wrapped.atom_data[:, :3] - whole.atom_data[:, :3]) @ np.linalg.inv(cell).T
        np.testing.assert_allclose(difference, np.rint(difference), atol=1e-6)
        self.assertTrue(np.all(np.abs(wrapped.atom_data[:, :3]) <= 5.))

    def test_box_follows_molecule_without_changing_molecular_zoom(self):
        without, _ = make_scene(self.model, scope=2, show_box=False)
        with_box, _ = make_scene(self.model, scope=2, show_box=True)
        self.assertEqual(len(without.box_vertices), 0)
        self.assertEqual(len(with_box.box_vertices), 24)
        self.assertEqual(without.radius, with_box.radius)
        np.testing.assert_array_equal(without.atom_data, with_box.atom_data)
        members = self.model.species[0].molecules[0]
        shift = ordered_positions(self.model, None)[members].mean(axis=0)
        np.testing.assert_allclose(with_box.box_vertices,
                                   box_line_vertices(self.model.scene.cell_matrix) - shift, atol=1e-6)

    def test_tilted_frame_cell_updates_box_and_preserves_source(self):
        cell = np.array([[12., 2., 1.], [0., 9., .5], [0., 0., 8.]])
        frame = DumpFrameData(0, 0, self.model.scene.atom_ids.copy(),
                              ordered_positions(self.model, None), cell,
                              np.ones(3, dtype=bool), False)
        original = frame.positions.copy()
        scene, _ = make_scene(self.model, frame, show_box=True)
        np.testing.assert_allclose(scene.box_vertices, box_line_vertices(cell))
        fractional = scene.box_vertices @ np.linalg.inv(cell).T
        np.testing.assert_allclose(np.abs(fractional), .5, atol=1e-6)
        self.assertEqual(len(np.unique(scene.box_vertices, axis=0)), 8)
        self.assertGreaterEqual(scene.radius,
                                np.linalg.norm(scene.box_vertices - scene.centre, axis=1).max() - 1e-6)
        np.testing.assert_array_equal(frame.positions, original)
        np.testing.assert_array_equal(frame.cell_matrix, cell)

    def test_no_cell_means_no_box(self):
        self.model.scene.cell_matrix = None
        self.model.scene.periodic = None
        scene, _ = make_scene(self.model, show_box=True)
        self.assertEqual(len(scene.box_vertices), 0)

    def test_hidden_and_dimmed_environment(self):
        hidden, _ = make_scene(self.model, scope=2)
        dimmed, _ = make_scene(self.model, scope=2, dim=True)
        self.assertEqual(len(hidden.atom_data), 3)
        self.assertEqual(len(dimmed.atom_data), 6)
        np.testing.assert_allclose(dimmed.atom_data[3:, 4:7], np.tile([.12, .15, .19], (3, 1)), atol=1e-6)

    def test_cg_mismatch_is_not_silently_assigned_to_species(self):
        with self.assertRaisesRegex(ValueError, 'does not match'):
            load_cg_model(self.path, self.model)

    def test_cg_labels_are_not_elements_and_keep_mapping_colors(self):
        path = Path(self.temp.name) / 'cg.data'
        for label in ('A', 'B1', 'Na', 'C'):
            with self.subTest(label=label):
                path.write_text('''CG beads

2 atoms
1 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 18.015 # {label}

Atoms # molecular

1 1 1 2 2 2
2 2 1 5 5 5
'''.format(label=label))
                with patch('openmscg_gui.trajectory_gpu._element_for_lammps_type',
                           side_effect=AssertionError('CG beads must not infer elements')):
                    model = load_cg_model(path, self.model)
                np.testing.assert_allclose(model.scene.atom_data[:, 3], 1.0)
                np.testing.assert_allclose(model.scene.atom_data[:, 4:7],
                                           [model_bead_colour(self.model, i) for i in range(2)])
                np.testing.assert_array_equal(model.scene.atom_ids, [1, 2])
                self.assertEqual(len(model.species[0].molecules), 2)

    def test_index_cancelled_and_cache_bounded(self):
        path = Path(self.temp.name) / 'water.dump'
        header = 'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id xs ys zs\n1 .5 .5 .5\n'
        path.write_text(''.join(header.format(step=i) for i in range(6)))
        with self.assertRaises(InterruptedError):
            LammpsDumpTrajectory(path).build_index(lambda: True)
        reader = LammpsDumpTrajectory(path).build_index()
        for i in range(6):
            frame = reader.load_frame(i)
            np.testing.assert_allclose(frame.positions, [[0, 0, 0]])
        self.assertEqual(len(reader._cache), 3)


if __name__ == '__main__':
    unittest.main()
