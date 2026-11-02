from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from openmscg_gui.backend import validation_metrics

from openmscg_gui.backend.validation_metrics import (
    load_property_statistics,
    _read_sampled_frames,
    validate_system,
)


class ValidationMetricsTests(unittest.TestCase):
    def test_identical_small_trajectories_have_zero_distribution_distances(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "output" / "work"
            lammps = root / "output" / "lammps"
            best = root / "optimization" / "best"
            work.mkdir(parents=True)
            lammps.mkdir(parents=True)
            best.mkdir(parents=True)

            structure = """small CG structure

6 atoms
4 bonds
2 angles

Atoms

1 1 1 1.0 1.0 1.0
2 1 1 2.0 1.0 1.0
3 1 1 3.0 1.0 1.0
4 2 1 6.0 6.0 6.0
5 2 1 7.0 6.0 6.0
6 2 1 8.0 6.0 6.0

Bonds

1 1 1 2
2 1 2 3
3 1 4 5
4 1 5 6

Angles

1 1 1 2 3
2 1 4 5 6
"""
            (lammps / "structure.data").write_text(structure)
            trajectory = """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
6
ITEM: BOX BOUNDS pp pp pp
0 12
0 12
0 12
ITEM: ATOMS id type mol x y z
1 1 1 1.0 1.0 1.0
2 1 1 2.0 1.0 1.0
3 1 1 3.0 1.0 1.0
4 1 2 6.0 6.0 6.0
5 1 2 7.0 6.0 6.0
6 1 2 8.0 6.0 6.0
"""
            (work / "types_cg.lammpstrj").write_text(trajectory)
            (best / "cg.lammpstrj").write_text(trajectory)

            def simulation_row(timestep, temperature, pressure, density, total_energy):
                values = [0.0] * 27
                values[0] = timestep
                values[2] = temperature
                values[3] = pressure
                values[5] = density
                values[9] = total_energy
                return " ".join(str(value) for value in values)

            aa_properties = root / "aa-source" / "simulation_data.out"
            aa_properties.parent.mkdir()
            aa_properties.write_text(
                simulation_row(100, 300.0, 1.0, 0.98, -20.0)
                + "\n"
                + simulation_row(200, 302.0, 1.2, 1.02, -18.0)
                + "\n"
            )
            (best / "simulation_data.out").write_text(
                simulation_row(100, 299.0, 0.8, 0.97, -21.0)
                + "\n"
                + simulation_row(200, 303.0, 1.4, 1.03, -17.0)
                + "\n"
            )

            metrics = validate_system(root, aa_properties_path=aa_properties)

            self.assertAlmostEqual(metrics["summary"]["rdf_distance"], 0.0)
            self.assertAlmostEqual(metrics["summary"]["bond_distance"], 0.0)
            self.assertAlmostEqual(metrics["summary"]["angle_distance"], 0.0)
            comparison = {row["label"]: row for row in metrics["property_comparison"]}
            self.assertAlmostEqual(comparison["Temperature"]["mapped_aa"]["mean"], 301.0)
            self.assertAlmostEqual(comparison["Temperature"]["mapped_aa"]["std"], 1.0)
            self.assertAlmostEqual(comparison["Temperature"]["cg"]["mean"], 301.0)
            self.assertAlmostEqual(comparison["Temperature"]["cg"]["std"], 2.0)
            self.assertEqual(
                set(comparison),
                {"Temperature", "Pressure", "Mass density", "Total energy"},
            )
            saved = json.loads((root / "validation" / "metrics.json").read_text())
            self.assertEqual(saved["topology"]["bonds"], 4)
            self.assertEqual(saved["sampling"]["cg_frames"], 1)

            merged = root / "merge" / "best" / "systems" / "small" / "cg.lammpstrj"
            merged.parent.mkdir(parents=True)
            merged.write_text(trajectory)
            merged_metrics = validate_system(root, merged_candidate=merged)
            self.assertEqual(merged_metrics["sources"]["cg_result"], "merged optimization")
            self.assertEqual(Path(merged_metrics["sources"]["cg"]), merged.resolve())

            baseline = lammps / "cg.lammpstrj"
            baseline.write_text(trajectory)
            baseline_metrics = validate_system(root, cg_candidate=baseline)
            self.assertEqual(baseline_metrics["sources"]["cg_result"], "baseline simulation")
            self.assertEqual(Path(baseline_metrics["sources"]["cg"]), baseline.resolve())
            # An explicitly chosen missing result must not fall back to older fits.
            with self.assertRaises(FileNotFoundError):
                validate_system(root, cg_candidate=lammps / "missing.lammpstrj")

            # Unequal lengths use the same fractional window, not a short tail
            # or shared absolute timesteps. Property-file means are unchanged.
            (work / 'types_cg.lammpstrj').write_text(''.join(
                trajectory.replace('\n100\n', '\n{}\n'.format(index * 100), 1)
                for index in range(20)))
            baseline.write_text(''.join(
                trajectory.replace('\n100\n', '\n{}\n'.format(index * 100), 1)
                for index in range(10)))
            (lammps / 'simulation_data.out').write_text((best / 'simulation_data.out').read_text())
            windowed = validate_system(root, cg_candidate=baseline, aa_properties_path=aa_properties,
                                       sampling_settings=dict(maximum_frames=7, discard_percent=20))
            sampling = windowed['sampling']
            self.assertEqual(sampling['method'], 'uniform_frame_indices')
            self.assertEqual(sampling['mapped_aa_frames'], 7)
            self.assertEqual(sampling['cg_frames'], 7)
            self.assertEqual(sampling['mapped_aa']['total_frames'], 20)
            self.assertEqual(sampling['mapped_aa']['discarded_frames'], 4)
            self.assertEqual(sampling['mapped_aa']['first_timestep'], 400)
            self.assertEqual(sampling['mapped_aa']['last_timestep'], 1900)
            self.assertEqual(sampling['cg']['discarded_frames'], 2)
            self.assertEqual(sampling['cg']['first_timestep'], 200)
            self.assertEqual(sampling['cg']['last_timestep'], 900)
            properties = {item['label']: item for item in windowed['property_comparison']}
            self.assertAlmostEqual(properties['Temperature']['mapped_aa']['mean'], 301.0)
            self.assertAlmostEqual(properties['Temperature']['cg']['mean'], 301.0)

    def test_ave_time_header_matches_only_requested_cg_properties(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aa-properties.out"
            path.write_text(
                "# TimeStep temperature density unrelated\n"
                "100 298.0 0.95 10.0\n"
                "200 302.0 1.05 20.0\n"
            )
            requested = [
                {"label": "Temperature", "column": "c_thermo_temp"},
                {"label": "Pressure", "column": "c_thermo_press"},
                {"label": "Mass density", "column": "v_sysdensity"},
            ]

            properties = load_property_statistics(path, requested)

            self.assertEqual(
                [item["column"] for item in properties],
                ["c_thermo_temp", "v_sysdensity"],
            )
            self.assertEqual(properties[0]["source_column"], "temperature")
            self.assertAlmostEqual(properties[0]["mean"], 300.0)
            self.assertAlmostEqual(properties[0]["std"], 2.0)


class SampledTrajectoryTests(unittest.TestCase):
    @staticmethod
    def dump(count, scaled=False):
        header = 'xs ys zs' if scaled else 'x y z'
        rows = '2 1 .4 .5 .6\n1 1 .2 .3 .4\n' if scaled else '2 1 4 5 6\n1 1 2 3 4\n'
        return ''.join(
            'ITEM: TIMESTEP\n{}\nITEM: NUMBER OF ATOMS\n2\n'
            'ITEM: BOX BOUNDS pp pp pp\n10 30\n20 40\n30 50\n'
            'ITEM: ATOMS id mol {}\n{}'.format(index * 10, header, rows)
            for index in range(count))

    def test_only_selected_coordinates_decoded_across_window_including_endpoints(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'cg.lammpstrj'
            # Binary seek offsets must also be correct for CRLF on Windows.
            path.write_bytes(self.dump(12).replace('\n', '\r\n').encode('utf-8'))
            with patch.object(validation_metrics, '_read_sampled_frame',
                              wraps=validation_metrics._read_sampled_frame) as decode:
                frames, sampling = _read_sampled_frames(path, dict(maximum_frames=5, discard_percent=25))
            self.assertEqual(decode.call_count, 5)
            self.assertEqual(sampling['frame_indices'], [3, 5, 7, 9, 11])
            self.assertEqual([frame['timestep'] for frame in frames], [30, 50, 70, 90, 110])
            self.assertEqual(frames[0]['ids'], [1, 2])
            self.assertEqual(frames[0]['positions'].tolist(), [[2., 3., 4.], [4., 5., 6.]])

    def test_default_covers_full_trajectory_and_short_inputs_use_all_frames(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'cg.lammpstrj'
            path.write_text(self.dump(300))
            frames, sampling = _read_sampled_frames(path)
            self.assertEqual(len(frames), 150)
            self.assertEqual(frames[0]['timestep'], 0)
            self.assertEqual(frames[-1]['timestep'], 2990)
            self.assertEqual(sampling['discarded_frames'], 0)
            path.write_text(self.dump(3, scaled=True))
            frames, sampling = _read_sampled_frames(path)
            self.assertEqual(sampling['frame_indices'], [0, 1, 2])
            self.assertEqual(frames[0]['positions'].tolist(), [[14., 26., 38.], [18., 30., 42.]])
            path.write_text(self.dump(1))
            frames, _ = _read_sampled_frames(path, dict(discard_percent=95))
            self.assertEqual(len(frames), 1)

    def test_incomplete_unsampled_frame_is_not_silently_ignored(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'cg.lammpstrj'
            # With a two-frame limit, only frames 0 and 2 are sampled. Remove
            # the final atom of frame 1; the indexing pass must still reject it.
            path.write_text(self.dump(3).replace('1 1 2 3 4\nITEM: TIMESTEP\n20\n',
                                               'ITEM: TIMESTEP\n20\n'))
            with self.assertRaisesRegex(ValueError, 'Incomplete atom row'):
                _read_sampled_frames(path, dict(maximum_frames=2))
            path.write_text('')
            with self.assertRaisesRegex(ValueError, 'No complete'):
                _read_sampled_frames(path)

    def test_nonfinite_selected_coordinates_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'cg.lammpstrj'
            path.write_text(self.dump(1).replace('2 1 4 5 6', '2 1 nan 5 6'))
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                _read_sampled_frames(path)

    def test_pair_only_rdf_matches_the_original_full_matrix_calculation(self):
        import numpy as np

        positions = np.asarray([[.1, 1., 1.], [9.9, 1., 1.], [3., 4., 5.], [4., 4., 5.]])
        frames = [dict(positions=positions, box=[10., 10., 10.])]
        molecules = np.asarray([1, 2, 3, 3])
        edges = np.linspace(0., 4.9, 101)
        upper = np.triu_indices(4, 1)
        allowed = molecules[upper[0]] != molecules[upper[1]]
        delta = positions[:, None, :] - positions[None, :, :]
        delta -= np.rint(delta / 10.) * 10.
        distances = np.sqrt(np.sum(delta[upper] ** 2, axis=1))[allowed]
        observed = np.histogram(distances, bins=edges)[0]
        shell = 4. * np.pi / 3. * (edges[1:] ** 3 - edges[:-1] ** 3)
        expected = np.count_nonzero(allowed) * shell / 1000.
        original = observed / expected
        actual = validation_metrics._rdf_for_frames(frames, np.arange(4), molecules, edges, np)
        np.testing.assert_allclose(actual, original)


if __name__ == "__main__":
    unittest.main()
