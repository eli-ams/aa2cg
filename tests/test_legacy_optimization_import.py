"""Legacy JSON history import is read-only and verifies original potentials."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openmscg_gui.optimization_history import StudyHistory, file_hash
from openmscg_gui.simulation_protocol import DEFAULT_PROTOCOL, SMOKE_MARKER, run_lines


class LegacyOptimizationImportTests(unittest.TestCase):
    def test_compatible_legacy_history_imports_without_changing_any_old_file(self):
        from openmscg_gui.backend.corrector import run, config
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, optimization = root / 'source', root / 'optimization'
            source.mkdir()
            optimization.mkdir()
            input_text = 'units real\n' + SMOKE_MARKER + '\n' + '\n'.join(run_lines(DEFAULT_PROTOCOL, 'id type'))
            structure = 'Original structure and masses\n'
            properties = [{'column': 'v_sysdensity', 'target': .81}]
            (source / 'input.data').write_text(input_text)
            (source / 'structure.data').write_text(structure)
            (source / 'Pair_A-A.table').write_text('A-A\nN 3\n\n1 3 1 1\n2 4 0 0\n3 5 0 0\n')
            for number in range(3):
                path = optimization / ('best' if number == 2 else 'runs/trial_{:04d}'.format(number))
                path.mkdir(parents=True)
                amplitude = .1 * number
                (path / 'input.data').write_text(input_text)
                (path / 'structure.data').write_text(structure)
                rows = ['A-A', 'N 3', '']
                for index, (r, energy) in enumerate(((3., 1.), (4., 0.), (5., 0.)), start=1):
                    rows.append('{} {} {:.8f} 0'.format(index, r, energy + amplitude * math.exp(-.5 * (r - 4.) ** 2)))
                (path / 'Pair_A-A.table').write_text('\n'.join(rows) + '\n')
                metadata = dict(trial=number, properties=properties, include_aa_cg_in_loss=False,
                    effective_simulation_protocol={key: value for key, value in DEFAULT_PROTOCOL.items() if key != 'ramp_ps'},
                    pair_corrections=[dict(table='Pair_A-A.table', amplitude=amplitude, center=4., sigma=1.)])
                (path / 'correction.json').write_text(json.dumps(metadata))
                (path / 'score.json').write_text(json.dumps(dict(score=3. - number)))
                (path / 'trial_record.json').write_text(json.dumps(dict(trial=number,
                    started='2026-10-28 10:00:0{}'.format(number), parameters={'Pair_A_A_amplitude': amplitude})))
            before = {p: file_hash(p) for p in optimization.rglob('*') if p.is_file()}
            distribution = lambda low, high: SimpleNamespace(low=low, high=high)
            optuna = SimpleNamespace(distributions=SimpleNamespace(FloatDistribution=distribution,
                distribution_to_json=lambda d: json.dumps(vars(d)), json_to_distribution=json.loads),
                trial=SimpleNamespace(TrialState=SimpleNamespace(COMPLETE=SimpleNamespace(name='COMPLETE'),
                    FAIL=SimpleNamespace(name='FAIL')), create_trial=lambda **kwargs: SimpleNamespace(**kwargs)))
            with patch.multiple(config, OPTIMIZATION_DIR=optimization, SOURCE_LAMMPS_DIR=source,
                    PROPERTIES=properties, SIMULATION_PROTOCOL={}, INCLUDE_AA_CG_IN_LOSS=False):
                run._validate_legacy_best()
                history = StudyHistory(optimization, 'fingerprint')
                run._import_legacy_history(history, optuna)
                history.reconcile()
                restored = []
                history.restore(SimpleNamespace(add_trial=restored.append), optuna)
                self.assertEqual(len(restored), 3)
                self.assertEqual(history.state['best_number'], 2)
                self.assertEqual(history.best_settings()['Pair_A-A.table']['amplitude'], .2)
                (source / 'Pair_A-A.table').write_text('A-A\nN 3\n\n1 3 9 1\n2 4 0 0\n3 5 0 0\n')
                with self.assertRaisesRegex(ValueError, 'Source pair potentials changed'):
                    run._validate_legacy_best()
            for path, digest in before.items():
                self.assertEqual(file_hash(path), digest, str(path))


if __name__ == '__main__':
    unittest.main()
