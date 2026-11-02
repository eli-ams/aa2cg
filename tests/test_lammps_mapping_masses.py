"""Export regression cases; no OpenMSCG or LAMMPS process is required."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openmscg_gui.backend.lammps_generation import _load_molecules, _validate_export_masses
from openmscg_gui.backend.openmscg_wrap.mapping import Bead, Molecule
from openmscg_gui.backend.openmscg_wrap.structure import masses_for_bead_types


class LammpsMappingMassTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.molecules = [Molecule(name='example', number=100, formula='C3H6',
            beads_per_molecule=3, beads=[
                Bead(name=label, molecule_index=0, local_index=index,
                     formula=formula, bead_type=label, atom_ids=[index + 1])
                for index, (label, formula) in enumerate([('A', 'CH3'), ('B', 'CH2'), ('C', 'CH')])])]

    def test_export_reuses_manual_custom_mapping_not_default_united_atom_rules(self):
        rules = [{'bead_type': 'A', 'atom_indices': [0, 1, 2, 3]}]
        workflow = SimpleNamespace(output_dir=self.root, input_dir=self.root,
            files={'structure_data': 'aa.data'}, molecules=[SimpleNamespace(
                name='example', number=100, formula='C3H6', smiles='CCC')],
            mapping_mode='manual_custom', custom_mapping_rules=rules)
        with patch('openmscg_gui.backend.openmscg_wrap.structure_detection.prepare_structure_mapping',
                   return_value=SimpleNamespace(molecules=self.molecules)) as prepare:
            self.assertEqual(_load_molecules(workflow), self.molecules)
        self.assertEqual(prepare.call_args[1]['mapping_mode'], 'manual_custom')
        self.assertEqual(prepare.call_args[1]['custom_mapping_rules'], rules)

    def test_export_keeps_automatic_mapping_when_requested(self):
        workflow = SimpleNamespace(output_dir=self.root, input_dir=self.root,
            files={'structure_data': 'aa.data'}, molecules=[],
            mapping_mode='united_atom', custom_mapping_rules=None)
        with patch('openmscg_gui.backend.openmscg_wrap.structure_detection.prepare_structure_mapping',
                   return_value=SimpleNamespace(molecules=[])) as prepare:
            _load_molecules(workflow)
        self.assertEqual(prepare.call_args[1]['mapping_mode'], 'united_atom')
        self.assertEqual(prepare.call_args[1]['custom_mapping_rules'], [])

    def test_squalane_bead_masses_follow_saved_labels(self):
        masses = masses_for_bead_types(self.molecules, ['A', 'B', 'C'])
        for actual, expected in zip(masses, [15.03452, 14.02658, 13.01864]):
            self.assertAlmostEqual(actual, expected)
        structure = self.root / 'structure.data'
        structure.write_text('CG\n\nMasses\n\n1 15.034520 # A\n'
                             '2 14.026580 # B\n3 13.018640 # C\n\nAtoms\n')
        _validate_export_masses(self.molecules, structure)

    def test_changed_mass_assignment_is_rejected_before_export(self):
        structure = self.root / 'structure.data'
        structure.write_text('CG\n\nMasses\n\n1 15.034520 # A\n'
                             '2 13.018640 # B\n3 14.026580 # C\n\nAtoms\n')
        with self.assertRaisesRegex(ValueError, 'do not match the built CG model'):
            _validate_export_masses(self.molecules, structure)

    def test_changed_type_order_is_rejected_even_if_total_mass_is_unchanged(self):
        structure = self.root / 'structure.data'
        structure.write_text('CG\n\nMasses\n\n1 15.034520 # A\n'
                             '2 13.018640 # C\n3 14.026580 # B\n\nAtoms\n')
        with self.assertRaisesRegex(ValueError, 'Rebuild the CG model'):
            _validate_export_masses(self.molecules, structure)

    def test_older_workflow_without_reference_remains_supported(self):
        _validate_export_masses(self.molecules, self.root / 'missing.data')


if __name__ == '__main__':
    unittest.main()
