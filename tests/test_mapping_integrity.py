from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openmscg_gui.backend.openmscg_wrap.structure import validate_structure_bonds
from openmscg_gui.backend.openmscg_wrap.structure_detection import (
    _components,
    _molecule_ids_from_inferred_bonds,
    _validate_atom_partition,
    read_lammps_structure,
)
from openmscg_gui.backend.openmscg_wrap.trajectory_mapping import labelled_graph_isomorphism
from openmscg_gui.backend.openmscg_wrap.validation import validate_trajectory_atom_ids
from openmscg_gui.backend.openmscg_wrap.trajectory_inspection import inspect_trajectory
from openmscg_gui.project import SystemSpec


class MappingIntegrityTests(unittest.TestCase):
    def test_old_detected_project_migrates_to_completed_inspection(self):
        system = SystemSpec.from_dict(
            {
                "detected_species": [{"id": "species-1", "count": 1}],
                "stages": {"map": "complete"},
            }
        )
        self.assertEqual(system.stages["inspect"], "complete")

    def test_trajectory_inspection_reports_frames_stride_and_fields(self):
        with TemporaryDirectory() as directory:
            trajectory = Path(directory) / "trajectory.lammpstrj"
            frames = []
            for timestep in (0, 50):
                frames.append(
                    f"""ITEM: TIMESTEP
{timestep}
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 11
0 12
ITEM: ATOMS id mol element x y z fx fy fz
1 1 C 1 1 1 0.1 0.2 0.3
2 1 H 2 1 1 -0.1 -0.2 -0.3
"""
                )
            trajectory.write_text("".join(frames))

            stats = inspect_trajectory(trajectory)

            self.assertEqual(stats["frames"], 2)
            self.assertEqual(stats["frame_stride"], 50)
            self.assertEqual(stats["timestep_span"], 50)
            self.assertTrue(stats["has_forces"])
            self.assertTrue(stats["has_molecule_ids"])
            self.assertTrue(all(check["status"] == "pass" for check in stats["checks"]))

    def test_complete_atom_partition_accepts_dictionary_keys(self):
        atoms = {1: object(), 2: object(), 3: object()}
        _validate_atom_partition([1, 2, 3], atoms.keys())

        with self.assertRaisesRegex(ValueError, r"unexpected atom IDs: \[4\]"):
            _validate_atom_partition([1, 2, 4], atoms.keys())

    def test_inferred_bonds_define_molecular_components_without_mol_ids(self):
        molecule_ids = _molecule_ids_from_inferred_bonds(
            [1, 2, 3, 4, 5], {(1, 2), (2, 3), (4, 5)}
        )
        self.assertEqual([molecule_ids[index] for index in range(1, 6)], [1, 1, 1, 2, 2])

    def test_topology_mapping_accepts_shuffled_dump_rows_but_requires_exact_ids(self):
        with TemporaryDirectory() as directory:
            trajectory = Path(directory) / "aa.lammpstrj"
            template = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
{rows}
"""
            trajectory.write_text(template.format(rows="2 1 1 0 0\n1 1 0 0 0"))
            validate_trajectory_atom_ids(trajectory, 2)

            trajectory.write_text(template.format(rows="3 1 1 0 0\n1 1 0 0 0"))
            with self.assertRaisesRegex(ValueError, "do not match structure.data"):
                validate_trajectory_atom_ids(trajectory, 2)

    def test_lammps_data_drives_molecule_membership_and_counts(self):
        with TemporaryDirectory() as directory:
            structure = Path(directory) / "atomistic.data"
            structure.write_text(
                """atomistic mixture

2 atom types
1 bond types

5 atoms
3 bonds

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 12.011 # C
2 1.008 # H

Atoms # molecular

1 1 1 0.0 0.0 0.0
2 1 2 1.0 0.0 0.0
3 2 1 5.0 0.0 0.0
4 2 2 6.0 0.0 0.0
5 2 2 5.0 1.0 0.0

Bonds

1 1 1 2
2 1 3 4
3 1 3 5
"""
            )

            parsed = read_lammps_structure(structure)
            components = _components(parsed)

            self.assertEqual(len(parsed.atoms), 5)
            self.assertEqual(len(parsed.bonds), 3)
            self.assertEqual([item.molecule_id for item in components], [1, 2])
            self.assertEqual([len(item.atom_ids) for item in components], [2, 3])

    def test_labelled_graph_isomorphism_recovers_permuted_atom_order(self):
        source_elements = ["H", "O", "C", "C"]
        source_adjacency = [{1}, {0, 2}, {1, 3}, {2}]
        target_elements = ["C", "H", "C", "O"]
        target_adjacency = [{2}, {3}, {0, 3}, {1, 2}]

        mapping = labelled_graph_isomorphism(
            source_elements,
            source_adjacency,
            target_elements,
            target_adjacency,
        )

        self.assertEqual({mapping[index] for index in range(4)}, set(range(4)))
        self.assertTrue(
            all(
                ((other in source_adjacency[index]) == (mapping[other] in target_adjacency[mapping[index]]))
                for index in range(4)
                for other in range(4)
            )
        )

    def test_structure_validation_rejects_long_bond(self):
        with TemporaryDirectory() as directory:
            structure = Path(directory) / "structure.data"
            structure.write_text(
                """bad structure

1 atom types
1 bond types

2 atoms
1 bonds

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 12.0 # C

Atoms

1 1 1 0.0 0.0 0.0
2 1 1 8.0 0.0 0.0

Bonds

1 1 1 2
"""
            )

            with self.assertRaisesRegex(ValueError, "does not align topology bonds"):
                validate_structure_bonds(structure)


if __name__ == "__main__":
    unittest.main()
