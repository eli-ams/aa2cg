from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


RAW_TABLE_DIR = Path("output/tables")
PROCESSED_TABLE_DIR = Path("output/processed_tables")

TRIM_TRIALS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
FIT_POINTS = [2, 3, 5, 7, 15, 21]
SMOOTH_WINDOWS = [5, 7, 9]


def delete_all_files_in_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {str(e)}")


def get_list_files_with_extension(directory="output/", extension=".table"):
    file_list = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            file_list.append(file)
    return file_list


class TableProcessor:
    def __init__(self, file: str, input_dir: str, output_dir: str):
        self.xi = None
        self.yi = None
        self.new_y_right = None
        self.new_y_left = None
        self.overall_accept = None
        self.accept_energies = False
        self.accept_forces = False
        self.right_intercept = None
        self.right_slope = None
        self.left_intercept = None
        self.left_slope = None
        self.interaction_type = None
        self.distances = None
        self.energies = None
        self.forces = None
        self.reason = None

        self.file = file
        self.input_dir = input_dir
        self.output_dir = output_dir

    def write_table_file_to_dir(self, output_dir, output_name):
        header = "".join(self.lines[0:3])
        header += f"N {len(self.distances)} # {round(max(self.distances), 8)} {round(max(self.energies), 8)} {round(max(self.forces), 8)}\n"
        str_output_file = "".join(header) + "\n"
        for i in range(len(self.distances)):
            str_output_file += f"{i + 1} {round(self.distances[i], 8)} {round(self.energies[i], 8)} {round(self.forces[i], 8)}\n"

        with open(os.path.join(output_dir, output_name), "w") as f:
            f.write(str_output_file)

    def int_or_float(self, str_entry):
        try:
            return int(str_entry)
        except:
            try:
                return float(str_entry)
            except:
                return str_entry

    def read_file(self):
        with open(os.path.join(self.input_dir, self.file), "r") as f:
            self.lines = f.readlines()
            data_lines = [
                [self.int_or_float(value) for value in line.split()]
                for line in self.lines[5:]
            ]

        try:
            data = [[row[i] for row in data_lines] for i in range(len(data_lines[0]))]
            self.distances, self.energies, self.forces = data[1], data[2], data[3]
        except IndexError as e:
            print(f"Error while reading the file: {e}")

        self.interaction_type = self.file.split("_")[0]

    def smoothen(self, target="energies", window_length=11, polyorder=3):
        if not window_length:
            return
        elif window_length % 2 == 0:
            window_length += 1
        array_to_smooth = getattr(self, target, None)
        if array_to_smooth is not None:
            smoothed = savgol_filter(array_to_smooth, window_length, polyorder)
            setattr(self, target, smoothed)

    def fit_cubic(self, target="energies"):
        cs = CubicSpline(self.distances, getattr(self, target))
        return cs

    def compute_derivative(self, cs):
        cs_1d = cs.derivative()
        self.forces = cs_1d(self.distances)

    def multiply_by_constant(self, constant, target="energies"):
        array_to_multiply = constant * getattr(self, target, None)
        setattr(self, target, array_to_multiply)

    def plot(self, file_out, target="energies"):
        plt.scatter(
            self.distances,
            getattr(self, target),
            label=f"{target.capitalize()} Data Points",
        )
        plt.legend()
        plt.xlabel("Distance")
        plt.ylabel(target.capitalize())
        plt.title(f"{target.capitalize()} vs Distance")
        plt.savefig(file_out)
        plt.clf()
        plt.close()

    def discard_forces(self):
        self.forces = None

    def downsample(self, ratio=1.0, target="energies"):
        if ratio <= 0 or ratio > 1:
            print("Invalid ratio. Must be between 0 and 1.")
            return

        step = int(1 / ratio)
        new_data = getattr(self, target, [])[::step]
        setattr(self, target, new_data)

    def extend_array(
        self,
        target,
        x_left,
        x_right,
        x_diff_tol=0.1,
        nbr_fit_points=5,
        slope_threshold=0,
    ):
        self.xi = np.array(self.distances)
        self.yi = np.array(getattr(self, target))

        if self.xi.size == 0 or self.yi.size == 0:
            print("Distances or target values not initialized.")
            return

        x_spacing = self.xi[1] - self.xi[0]
        x_spacing = max(x_spacing, x_diff_tol)

        new_x_left = np.arange(self.xi[0] - x_spacing, x_left - x_spacing, -x_spacing)[
            ::-1
        ]
        new_x_right = np.arange(self.xi[-1] + x_spacing, x_right + x_spacing, x_spacing)

        if new_x_left.size > 0:
            new_x_left[0] = x_left
        if new_x_right.size > 0:
            new_x_right[-1] = x_right

        if self.interaction_type in ["Bond", "Angle"] and target == "forces":
            slope, intercept = np.polyfit(self.xi, self.yi, 1)
            self.left_slope = self.right_slope = slope
            self.left_intercept = self.right_intercept = intercept
        else:
            self.left_slope, self.left_intercept = np.polyfit(
                self.xi[:nbr_fit_points], self.yi[:nbr_fit_points], 1
            )
            self.right_slope, self.right_intercept = np.polyfit(
                self.xi[-nbr_fit_points:], self.yi[-nbr_fit_points:], 1
            )

        if abs(self.left_slope) < slope_threshold:
            print("SLOPE ERROR!")
            last_x, last_y = self.xi[nbr_fit_points - 1], self.yi[nbr_fit_points - 1]
            self.left_slope = np.sign(self.left_slope) * slope_threshold
            self.left_intercept = last_y - (self.left_slope * last_x)

        self.new_y_left = self.left_slope * new_x_left + self.left_intercept
        self.new_y_right = self.right_slope * new_x_right + self.right_intercept

        final_x = np.concatenate([new_x_left, self.xi, new_x_right])
        final_y = np.concatenate([self.new_y_left, self.yi, self.new_y_right])

        self.distances = final_x
        setattr(self, target, final_y)

    def apply_damping(
        self, target="energies", decay_constant=0.1, damping_window_percent=20
    ):
        array_to_damp = np.array(getattr(self, target, None))
        if array_to_damp is None:
            print(f"Target {target} not found.")
            return

        damping_window_size = int(len(array_to_damp) * (damping_window_percent / 100))

        if damping_window_size == 0:
            print(
                "Damping window size calculated as zero based on the provided percentage."
            )
            return

        damping_start_idx = len(array_to_damp) - damping_window_size
        x_values = np.linspace(0, 1, damping_window_size)
        damping_factors = np.exp(-decay_constant * x_values)
        array_to_damp[damping_start_idx:] *= damping_factors
        setattr(self, target, array_to_damp)

    def validate_energies(self):
        if self.interaction_type == "Pair":
            if self.left_slope > 0:
                self.accept_energies = False
                self.reason = "ls+0"
                return

            if abs(self.left_slope) < 0.05:
                self.accept_energies = False
                self.reason = "absls-0.05"
                return

            if abs(self.right_slope) > 0.2:
                self.accept_energies = False
                self.reason = "rs+0.2"
                return

            if all(x < 0 for x in self.new_y_left):
                self.accept_energies = False
                self.reason = "ls--"
                return

            if max(self.new_y_left) < max(self.yi):
                self.accept_energies = False
                self.reason = "ls<ctr"
                return

        elif self.interaction_type in ["Bond", "Angle"]:
            if self.left_slope > 0:
                self.accept_energies = False
                self.reason = "ls+0"
                return
            if self.right_slope < 0:
                self.accept_energies = False
                self.reason = "rs-0"
                return
            if self.left_slope * self.right_slope > 0:
                self.reason = "ls.rs+0"
                self.accept_energies = False
                return

        if not self.has_root(target="energies"):
            self.accept_energies = False
            self.reason = "noroot"
            return

        self.accept_energies = True

    def validate_forces(self):
        if self.interaction_type == "Pair":
            if self.left_slope > 0:
                self.accept_forces = False
                self.reason = "ls+0"
                return

            if abs(self.left_slope) < 0.05:
                self.accept_forces = False
                self.reason = "absls-0.5"
                return

            if abs(self.right_slope) > 0.2:
                self.accept_forces = False
                self.reason = "rs+0.2"
                return

            if all(x < 0 for x in self.new_y_left):
                self.accept_energies = False
                self.reason = "ls--"
                return

            if max(self.new_y_left) < max(self.yi):
                self.accept_forces = False
                self.reason = "ls<ctr"
                return

        elif self.interaction_type in ["Bond", "Angle"]:
            if self.left_slope > 0:
                self.accept_forces = False
                self.reason = "ls+0"
                return
            if self.right_slope > 0:
                self.accept_forces = False
                self.reason = "rs+0"
                return
            if self.left_slope * self.right_slope < 0:
                self.accept_forces = False
                self.reason = "ls.rs-0"
                return

        if not self.has_root(target="forces"):
            self.accept_forces = False
            self.reason = "noroot"
            return

        self.accept_forces = True

    def trim_ends(self, fraction):
        if fraction < 0 or fraction >= 0.5:
            print("Invalid fraction. Must be between 0 and 0.5.")
            return

        targets = ["distances", "energies"]
        trimmed_data = {}

        for target in targets:
            if getattr(self, target, None) is None:
                print(f"Target {target} not found.")
                return

        for target in targets:
            array_to_trim = getattr(self, target)
            n = len(array_to_trim)
            cut = int(n * fraction)
            trimmed_array = array_to_trim[cut : n - cut]
            trimmed_data[target] = trimmed_array

        for target, trimmed_array in trimmed_data.items():
            setattr(self, target, trimmed_array)

    def has_root(self, target, threshold=0.25):
        yi = np.array(getattr(self, target))

        if yi.size == 0:
            print(f"{target} values not initialized.")
            return False

        zero_crossing_indices = np.where(
            (yi[:-1] * yi[1:] < 0)
            | ((np.abs(yi[:-1]) < threshold) & (np.abs(yi[1:]) > threshold))
            | ((np.abs(yi[1:]) < threshold) & (np.abs(yi[:-1]) > threshold))
        )[0]

        return len(zero_crossing_indices) > 0

    def export_files(self):

        folder = os.path.join(self.output_dir, "1_all")
        os.makedirs(folder, exist_ok=True)

        self.plot(os.path.join(folder, f"{self.file}.energies.png"), target="energies")

        self.plot(os.path.join(folder, f"{self.file}.forces.png"), target="forces")

        self.write_table_file_to_dir(output_dir=folder, output_name=self.file)

    def export_accepted_files(self):

        folder = os.path.join(self.output_dir, "2_accepted")
        os.makedirs(folder, exist_ok=True)

        self.write_table_file_to_dir(output_dir=folder, output_name=self.file)

        self.plot(
            os.path.join(folder, f"{self.file}.energies.png"),
            target="energies",
        )

        self.plot(os.path.join(folder, f"{self.file}.forces.png"), target="forces")

    def export_rejected_files(self, subdir="2_rejected"):

        folder = os.path.join(self.output_dir, subdir)
        os.makedirs(folder, exist_ok=True)

        self.write_table_file_to_dir(
            output_dir=folder,
            output_name=f"{self.file}.{self.reason}",
        )

        self.plot(
            os.path.join(folder, f"{self.file}.{self.reason}.energies.png"),
            target="energies",
        )

        self.plot(
            os.path.join(folder, f"{self.file}.{self.reason}.forces.png"),
            target="forces",
        )

    def run(self, trim_value, fit_point, smooth_window, extension_bounds=None):
        if self.file:

            self.read_file()
            self.discard_forces()

            if self.interaction_type == "Dihedral":
                # Dihedral tables already span their complete -180..180 degree
                # domain. Do not trim or linearly extend their periodic range.
                self.smoothen(
                    target="energies",
                    window_length=smooth_window,
                    polyorder=3,
                )
                cse = self.fit_cubic(target="energies")
                self.compute_derivative(cse)
                self.multiply_by_constant(target="forces", constant=-1)
                force_window = min(6 * smooth_window + 1, len(self.forces))
                if force_window % 2 == 0:
                    force_window -= 1
                if force_window >= 5:
                    self.smoothen(
                        target="forces",
                        window_length=force_window,
                        polyorder=2,
                    )
                finite = bool(
                    np.all(np.isfinite(self.distances))
                    and np.all(np.isfinite(self.energies))
                    and np.all(np.isfinite(self.forces))
                )
                self.accept_energies = finite
                self.accept_forces = finite
                self.overall_accept = finite
                self.reason = None if finite else "nonfinite"
                return

            left, right = extension_bounds or default_processing_settings(self.interaction_type)['bounds']
            self.trim_ends(fraction=trim_value)

            self.smoothen(target="energies", window_length=smooth_window, polyorder=3)

            cse = self.fit_cubic(target="energies")

            self.compute_derivative(cse)
            self.multiply_by_constant(target="forces", constant=-1)
            self.smoothen(
                target="forces", window_length=6 * smooth_window + 1, polyorder=2
            )

            store_x = copy.deepcopy(self.distances)

            if self.interaction_type == "Pair":
                self.apply_damping(
                    target="energies", decay_constant=2.5, damping_window_percent=10
                )
                self.extend_array(
                    target="energies",
                    x_left=left,
                    x_right=right,
                    x_diff_tol=0.1,
                    nbr_fit_points=fit_point,
                )

            elif self.interaction_type == "Bond":
                self.extend_array(
                    target="energies",
                    x_left=left,
                    x_right=right,
                    x_diff_tol=0.1,
                    nbr_fit_points=fit_point,
                )
            else:
                self.extend_array(
                    target="energies",
                    x_left=left,
                    x_right=right,
                    x_diff_tol=2.5,
                    nbr_fit_points=fit_point,
                )

            self.validate_energies()

            self.distances = store_x

            if self.interaction_type == "Pair":
                self.apply_damping(
                    target="forces", decay_constant=2.5, damping_window_percent=10
                )
                self.extend_array(
                    target="forces",
                    x_left=left,
                    x_right=right,
                    x_diff_tol=0.1,
                    nbr_fit_points=fit_point,
                )
                self.apply_damping(
                    target="forces", decay_constant=10, damping_window_percent=15
                )

            elif self.interaction_type == "Bond":
                self.extend_array(
                    target="forces",
                    x_left=left,
                    x_right=right,
                    x_diff_tol=0.1,
                    nbr_fit_points=fit_point,
                )
            else:
                self.extend_array(
                    target="forces",
                    x_left=left,
                    x_right=right,
                    x_diff_tol=2.5,
                    nbr_fit_points=fit_point,
                )

            self.validate_forces()

            if self.accept_energies and self.accept_forces:
                self.overall_accept = True
            else:
                self.overall_accept = False

        else:
            raise FileNotFoundError(".table files not found!")


def default_processing_settings(kind):
    bounds = {'Pair': (.5, 8.), 'Bond': (.5, 4.), 'Angle': (0., 180.),
              'Dihedral': (-180., 180.)}
    if kind not in bounds:
        raise ValueError('Unsupported table interaction kind: ' + kind)
    return dict(trim=TRIM_TRIALS[0], smooth=SMOOTH_WINDOWS[0], fit_points=FIT_POINTS[0],
                bounds=bounds[kind])


def normalize_processing_settings(kind, settings, distances=None):
    """Validate custom recipes without processing or exporting any table."""
    if not isinstance(settings, dict):
        raise ValueError('Invalid table-processing settings.')
    defaults = default_processing_settings(kind)
    try:
        trim = float(settings.get('trim', defaults['trim']))
        smooth = float(settings.get('smooth', defaults['smooth']))
        fit = float(settings.get('fit_points', defaults['fit_points']))
        left, right = map(float, settings.get('bounds', defaults['bounds']))
    except (ValueError, TypeError):
        raise ValueError('Enter numeric table-processing settings.')
    if not all(math.isfinite(v) for v in (trim, smooth, fit, left, right)):
        raise ValueError('Table-processing settings must be finite.')
    if not smooth.is_integer() or smooth < 5 or smooth > 1001 or int(smooth) % 2 != 1:
        raise ValueError('Smoothing must be an odd window from 5 to 1001 points.')
    if not 0 <= trim < .5:
        raise ValueError('Trim must be at least 0% and less than 50% per end.')
    if not fit.is_integer() or not 2 <= fit <= 10000:
        raise ValueError('Tail fitting requires 2–10,000 points.')
    if kind == 'Dihedral':
        # No trimming or tail extrapolation for periodic tables.
        trim, fit, (left, right) = 0., defaults['fit_points'], defaults['bounds']
    elif left < 0 or right <= left or (kind == 'Pair' and left == 0) or (kind == 'Angle' and right > 180):
        raise ValueError('Extension endpoints must be ordered and nonnegative (angles: 0–180°).')
    if distances is not None:
        x = np.asarray(distances, dtype=float)
        if len(x) < 5 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError('The raw table needs at least five finite, increasing coordinates.')
        cut = int(len(x) * trim)
        kept = x[cut:len(x) - cut]
        required = int(smooth) if kind == 'Dihedral' else max(6 * int(smooth) + 1, int(fit))
        if len(kept) < required:
            raise ValueError('Too few retained points: need at least {} for this smoothing/tail fit.'.format(required))
        if kind != 'Dihedral':
            if kind == 'Pair' and left >= kept[0]:
                raise ValueError('Pair tables need a left extension below the retained range for the existing boundary checks.')
            step = max(float(kept[1] - kept[0]), 2.5 if kind == 'Angle' else .1)
            if len(kept) + max(0, (kept[0] - left) / step) + max(0, (right - kept[-1]) / step) > 100000:
                raise ValueError('Extension would exceed 100,000 table points; reduce its range.')
    return dict(trim=trim, smooth=int(smooth), fit_points=int(fit), bounds=[left, right])


def processing_attempts(file, input_dir, output_dir, override=None, cancelled=None):
    """Shared calculation path for preview and export; no writes in this generator."""
    kind = file.split('_')[0]
    if override is not None:
        source = TableProcessor(file, input_dir, output_dir)
        source.read_file()
        recipes = [normalize_processing_settings(kind, override, source.distances)]
    else:
        defaults = default_processing_settings(kind)
        recipes = (dict(trim=trim, smooth=smooth, fit_points=fit, bounds=list(defaults['bounds']))
                   for trim in TRIM_TRIALS for smooth in SMOOTH_WINDOWS for fit in FIT_POINTS)
    for recipe in recipes:
        if cancelled is not None and cancelled():
            raise InterruptedError('Processing preview superseded.')
        processor = TableProcessor(file, input_dir, output_dir)
        processor.run(recipe['trim'], recipe['fit_points'], recipe['smooth'],
                      extension_bounds=recipe['bounds'])
        yield processor, recipe
        if processor.overall_accept:
            break


def preview_table(path, override=None, cancelled=None):
    """Fresh raw inputs, same search/checks as export, but no plots/files written."""
    path = Path(path)
    result = None
    for processor, recipe in processing_attempts(path.name, str(path.parent), '', override, cancelled):
        result = processor, recipe
    if result is None:
        raise ValueError('No processing candidate available.')
    processor, recipe = result
    arrays = [np.asarray(values, dtype=float).copy() for values in
              (processor.distances, processor.energies, processor.forces)]
    if len({len(a) for a in arrays}) != 1 or not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('Processing produced invalid or nonfinite table values.')
    return dict(curves=arrays, recipe=recipe, accepted=bool(processor.overall_accept), reason=processor.reason)


def main(overrides=None):
    working_dir = str(RAW_TABLE_DIR)
    if not RAW_TABLE_DIR.is_dir():
        raise FileNotFoundError(f"Raw table directory not found: {RAW_TABLE_DIR}")

    list_of_files = get_list_files_with_extension(
        directory=working_dir, extension=".table"
    )
    if not list_of_files:
        raise FileNotFoundError(f"No .table files found in {RAW_TABLE_DIR}")

    overrides = overrides or {}
    if not isinstance(overrides, dict):
        raise ValueError('Table overrides must be a dictionary.')
    # Validate custom settings before clearing any previous exported tables.
    for file in list_of_files:
        if file in overrides:
            source = TableProcessor(file, working_dir, '')
            source.read_file()
            normalize_processing_settings(source.interaction_type, overrides[file], source.distances)

    output_path = str(PROCESSED_TABLE_DIR)

    for subdir in [
        "1_all",
        "2_accepted",
        "2_rejected",
        "2_finally_rejected",
    ]:
        full_dir = os.path.join(output_path, subdir)
        os.makedirs(full_dir, exist_ok=True)
        delete_all_files_in_directory(full_dir)

    print(list_of_files)

    nbr_of_accepted_files = 0
    nbr_of_processed_files = 0
    chosen_settings = {}

    for file in list_of_files:
        accepted = False
        processor = None
        for processor, recipe in processing_attempts(file, working_dir, output_path, overrides.get(file)):
            processor.export_files()
            chosen_settings[file] = dict(recipe, accepted=bool(processor.overall_accept), reason=processor.reason)
            if processor.overall_accept:
                processor.export_accepted_files()
                nbr_of_accepted_files += 1
                accepted = True
            else:
                processor.export_rejected_files()
        if not accepted:
            if processor is None:
                raise RuntimeError(f"File {file} was not processed.")
            processor.export_rejected_files(subdir="2_finally_rejected")
            print(f"File {file} could not be optimized.")
        nbr_of_processed_files += 1
        print(f"Processing file {file}...")

    print(
        f"Finalized! {nbr_of_accepted_files} out of {nbr_of_processed_files} files accepted."
    )
    (PROCESSED_TABLE_DIR / 'processing_settings.json').write_text(
        json.dumps(chosen_settings, indent=2) + '\n', encoding='utf-8')


if __name__ == "__main__":
    main()
