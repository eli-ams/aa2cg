"""Defaults for bundled multi-system merging and global optimization.

The GUI worker supplies explicit completed-system paths and project-local output
folders before executing this module.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path.cwd()

INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = BASE_DIR / "combined_forcefield"
RUNS_DIR = BASE_DIR / "runs"
BEST_DIR = BASE_DIR / "best"

# If SYSTEMS is empty, folders in INPUT_DIR are discovered automatically.
# Each entry can contain: name, path, weight.
SYSTEMS = []

# Candidate subfolders searched inside each system folder.
LAMMPS_SOURCE_CANDIDATES = [
    Path("."),
    Path("best"),
    Path("lammps"),
    Path("output/lammps"),
]

# Required label standardization file.
# It is produced by Stage 2 and copied into the Stage 4 LAMMPS setup.
BEAD_SIGNATURE_FILE = "bead_signature.json"

# Deterministic merge settings.
GRID_TOLERANCE = 1.0e-7
CONFLICT_RMS_ENERGY = 2.0
CONFLICT_RMS_FORCE = 10.0
MERGE_FORCE_FROM_ENERGY = True

# This is run inside each system folder in each Optuna trial.
LAMMPS_COMMAND = ["mpiexec", "-np", "16", "lmp", "-in", "input.data"]
SIMULATION_DATA_FILE = "simulation_data.out"

# Optuna settings.
RUN_OPTIMIZER = True
STUDY_NAME = "merged_forcefield_correction"
N_TRIALS = 30
RANDOM_SEED = 12345

# Global Gaussian pair-table correction.
# U_new(r) = U_old(r) + amplitude * exp(-0.5 * ((r - center) / sigma)^2)
GAUSSIAN_AMPLITUDE_MIN = -2.0
GAUSSIAN_AMPLITUDE_MAX = 2.0
GAUSSIAN_CENTER_MIN = 2.5
GAUSSIAN_CENTER_MAX = 8.0
GAUSSIAN_SIGMA_MIN = 0.2
GAUSSIAN_SIGMA_MAX = 2.0

# Multi-property scoring from simulation_data.out.
# Each score is normalized by the tolerances below, then multiplied by weight.
# target=None means the first sampled value is used as the reference.
PROPERTIES = [
    {
        "column": "v_sysdensity",
        "target": None,
        "weight": 1.0,
        "tail_scale": 0.05,
        "drift_scale": 0.05,
        "std_scale": 0.05,
        "slope_scale": 0.05,
        "skip_fraction": 0.0,
        "min": 0.2,
        "max": 2.5,
        "max_abs_drift": 0.5,
        "hard_penalty": 1.0e6,
    },
    # Example:
    # {
    #     "column": "v_pe",
    #     "target": None,
    #     "weight": 0.3,
    #     "tail_scale": 200.0,
    #     "drift_scale": 200.0,
    #     "std_scale": 100.0,
    #     "slope_scale": 200.0,
    #     "skip_fraction": 0.0,
    #     "min": None,
    #     "max": None,
    #     "max_abs_drift": None,
    #     "hard_penalty": 1.0e6,
    # },
]

# Keep Optuna from solving the score by making violent table edits.
REGULARIZATION_WEIGHT = 0.05
