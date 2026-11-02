"""Protocol preparation/persistence regressions; no LAMMPS process is launched."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openmscg_gui.project import SystemSpec
from openmscg_gui.simulation_protocol import (DEFAULT_PROTOCOL, SMOKE_MARKER, normalize_protocol,
    run_lines, read_protocol, prepare_trial_input, validate_scoring_window, protocol_summary,
    ramp_stages)


def generated_input(**values):
    return ('units real\natom_style molecular\nread_data structure.data\n'
            'pair_style table spline 1000\n' + SMOKE_MARKER + '\n' +
            '\n'.join(run_lines(dict(DEFAULT_PROTOCOL, **values), 'id type mol x y z')) + '\n')


class SimulationProtocolTests(unittest.TestCase):
    def test_current_generator_is_supported_and_defaults_are_unchanged(self):
        from openmscg_gui.backend.lammps_generation import _simulation_lines
        for style in ('atomic', 'molecular', 'full'):
            text = 'units real\n' + '\n'.join(_simulation_lines(style)) + '\n'
            self.assertEqual(read_protocol(text)[0], DEFAULT_PROTOCOL)
            self.assertEqual(prepare_trial_input(text, {}), text)
            self.assertEqual(read_protocol(prepare_trial_input(text, DEFAULT_PROTOCOL))[0], DEFAULT_PROTOCOL)

    def test_duration_and_state_point_change_without_rewriting_setup(self):
        text = generated_input()
        overrides = dict(DEFAULT_PROTOCOL, production_steps=2_000_000, timestep_fs=.5,
                         temperature_k=310., pressure_atm=2.)
        result = prepare_trial_input(text, overrides)
        self.assertEqual(result.split(SMOKE_MARKER)[0], text.split(SMOKE_MARKER)[0])
        self.assertIn('velocity all create 310.0 1234', result)
        self.assertIn('fix npt all npt temp 310.0 310.0 100.0 iso 2.0 2.0 1000.0', result)
        self.assertIn('timestep 0.5', result)
        self.assertIn('run 2000000 post no', result)
        self.assertIn('fix simulation_data all ave/time 1 99 100', result)
        self.assertEqual(read_protocol(result)[0], overrides)
        self.assertIn('sample 1 ns', protocol_summary(overrides))

    def test_warmup_precedes_all_sampling_without_reinitializing_velocities(self):
        result = prepare_trial_input(generated_input(), {'equilibration_steps': 20_000})
        self.assertLess(result.index('run 20000 post no'), result.index('fix simulation_data'))
        self.assertLess(result.index('reset_timestep 0'), result.index('dump trajectories'))
        self.assertEqual(result.count('velocity all create'), 1)
        self.assertEqual(result.count('fix npt all npt'), 2)
        self.assertEqual(read_protocol(result)[0]['equilibration_steps'], 20_000)
        # Reapplying is deterministic, not a second inserted equilibration block.
        self.assertEqual(prepare_trial_input(result, {'equilibration_steps': 20_000}), result)

    def test_ramp_is_nvt_with_three_timesteps_then_unchanged_production(self):
        result = prepare_trial_input(generated_input(), {'ramp_ps': 12.})
        self.assertIn('fix ramp all nvt temp 298.15 298.15 100.0', result)
        for dt, steps in ((.25, 16000), (.5, 8000), (1., 4000)):
            self.assertIn('timestep {}\nrun {} post no'.format(dt, steps), result)
        self.assertLess(result.index('unfix ramp'), result.index('fix npt all npt'))
        self.assertLess(result.index('reset_timestep 0'), result.index('fix simulation_data'))
        self.assertEqual(result.count('velocity all create'), 1)
        self.assertEqual(result.count('dump trajectories'), 1)
        self.assertIn('run 500000 post no', result)
        self.assertEqual(read_protocol(result)[0], dict(DEFAULT_PROTOCOL, ramp_ps=12.))
        self.assertEqual(prepare_trial_input(result, {'ramp_ps': 12.}), result)

    def test_ramp_with_npt_settling_roundtrips_and_does_not_change_scoring_window(self):
        overrides = dict(DEFAULT_PROTOCOL, ramp_ps=10.123456789, timestep_fs=.7,
                         equilibration_steps=2500, production_steps=400)
        result = prepare_trial_input(generated_input(), overrides)
        self.assertEqual(read_protocol(result)[0], overrides)
        self.assertLess(result.index('unfix ramp'), result.index('run 2500 post no'))
        self.assertLess(result.index('run 2500 post no'), result.index('fix simulation_data'))
        validate_scoring_window(overrides, [{'column': 'density', 'skip_fraction': .5}])
        no_ramp = prepare_trial_input(result, {'ramp_ps': 0.})
        self.assertNotIn('fix ramp', no_ramp)
        self.assertIn('run 2500 post no', no_ramp)
        self.assertEqual(read_protocol(no_ramp)[0]['ramp_ps'], 0.)

    def test_unknown_commands_inside_ramp_are_not_rewritten(self):
        text = generated_input(ramp_ps=10.)
        text = text.replace('unfix ramp', 'unfix ramp\nfix extra all nve')
        self.assertEqual(prepare_trial_input(text, {}), text)
        with self.assertRaises(ValueError):
            prepare_trial_input(text, {'ramp_ps': 20.})

    def test_step_ramp_roundtrips_with_exact_total_and_timestep_derived_duration(self):
        for total in (3, 4, 5, 6, 7, 28_000, 46_667, 2_000_000_000):
            settings = normalize_protocol({'ramp_steps': total})
            stages = ramp_stages(settings)
            self.assertEqual(sum(steps for steps, _ in stages), total)
            self.assertTrue(all(steps > 0 for steps, _ in stages))
            result = prepare_trial_input(generated_input(), settings)
            self.assertEqual(read_protocol(result)[0], settings)
            self.assertEqual(prepare_trial_input(result, settings), result)
        original = normalize_protocol({'ramp_steps': 28_000})
        shorter = dict(original, timestep_fs=.5)
        self.assertIn('Ramp 12 ps', protocol_summary(original))
        self.assertIn('Ramp 6 ps', protocol_summary(shorter))
        self.assertEqual([steps for steps, _ in ramp_stages(original)],
                         [steps for steps, _ in ramp_stages(shorter)])

    def test_explicit_steps_replace_legacy_ramp_and_can_be_disabled(self):
        text = generated_input(ramp_ps=12.)
        self.assertEqual(prepare_trial_input(text, {}), text)
        result = prepare_trial_input(text, {'ramp_steps': 28_000})
        self.assertIn('# Workbench ramp-up steps: 28000', result)
        self.assertNotIn('# Workbench ramp-up ps:', result)
        self.assertEqual(read_protocol(result)[0]['ramp_steps'], 28_000)
        for off in ({'ramp_steps': 0}, dict(DEFAULT_PROTOCOL)):
            self.assertNotIn('fix ramp', prepare_trial_input(result, off))
        for values in ({'ramp_steps': 1}, {'ramp_steps': 2}, {'ramp_steps': 3.5},
                       {'ramp_steps': -1}, {'ramp_steps': 2_000_000_001},
                       {'ramp_steps': 100, 'ramp_ps': 12.}):
            with self.assertRaises(ValueError):
                normalize_protocol(values)

    def test_custom_inputs_are_preserved_or_rejected_not_silently_rewritten(self):
        for text in ('units metal\nrun 20000\n',
                     generated_input().replace('run 500000 post no', 'run ${steps}'),
                     generated_input().replace('ave/time 1 99 100', 'ave/time 1 9 10'),
                     generated_input().replace('timestep 1.0', 'fix extra all langevin 300 300 100 1234\ntimestep 1.0')):
            self.assertEqual(prepare_trial_input(text, {}), text)
            with self.assertRaises(ValueError):
                prepare_trial_input(text, DEFAULT_PROTOCOL)

    def test_inherited_nondefault_duration_damping_and_seed_are_preserved(self):
        text = generated_input(production_steps=20_000, timestep_fs=.25)
        text = text.replace('298.15 100.0 iso', '298.15 80.0 iso').replace('1234 rot no', '9876 rot no')
        result = prepare_trial_input(text, {'pressure_atm': 2.})
        settings, _, extra = read_protocol(result)
        self.assertEqual(settings['production_steps'], 20_000)
        self.assertEqual(settings['timestep_fs'], .25)
        self.assertEqual(extra['temperature_damping'], 80.)
        self.assertEqual(extra['seed'], 9876)

    def test_validation_and_property_sample_counts(self):
        for change in ({'production_steps': 199}, {'production_steps': 500.5},
                       {'equilibration_steps': -1}, {'timestep_fs': 0},
                       {'temperature_k': float('nan')}, {'pressure_atm': float('inf')}):
            with self.assertRaises(ValueError):
                normalize_protocol(change)
        for values in ({'ramp_ps': -1.}, {'ramp_ps': float('nan')},
                       {'ramp_ps': 1.e6, 'timestep_fs': 1.e-6}):
            with self.assertRaises(ValueError):
                normalize_protocol(values)
        properties = [{'column': 'v_sysdensity', 'skip_fraction': .5}]
        validate_scoring_window(dict(DEFAULT_PROTOCOL, production_steps=400), properties)
        with self.assertRaises(ValueError):
            validate_scoring_window(dict(DEFAULT_PROTOCOL, production_steps=200), properties)

    def test_legacy_and_per_system_round_trip(self):
        self.assertEqual(SystemSpec.from_dict({}).optimization_protocol, {})
        system = SystemSpec(optimization_protocol=dict(DEFAULT_PROTOCOL, production_steps=1_000_000))
        restored = SystemSpec.from_dict(asdict(system))
        self.assertEqual(restored.optimization_protocol, system.optimization_protocol)
        self.assertEqual(SystemSpec().optimization_protocol, {})

    def test_trial_changes_only_its_copy_and_records_effective_protocol(self):
        from openmscg_gui.backend.corrector import run, config
        import json
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            original = generated_input()
            (source / 'input.data').write_text(original, encoding='utf-8')
            overrides = dict(DEFAULT_PROTOCOL, production_steps=1_000_000)
            with patch.multiple(config, SOURCE_LAMMPS_DIR=source, WORK_DIR=root / 'runs',
                                SIMULATION_PROTOCOL=overrides), \
                 patch.object(run, '_suggest_and_apply_pair_corrections', return_value=[]), \
                 patch.object(run, '_initial_gaussians', return_value={}), \
                 patch.object(run, '_run_and_score', return_value=0.), \
                 patch.object(run.subprocess, 'run', side_effect=AssertionError('No LAMMPS in this test')):
                self.assertEqual(run.objective(SimpleNamespace(number=0)), 0.)
            self.assertEqual((source / 'input.data').read_text(encoding='utf-8'), original)
            trial = root / 'runs' / 'trial_0000'
            self.assertEqual(read_protocol((trial / 'input.data').read_text())[0], overrides)
            metadata = json.loads((trial / 'correction.json').read_text())
            self.assertEqual(metadata['effective_simulation_protocol'], overrides)


if __name__ == '__main__':
    unittest.main()
