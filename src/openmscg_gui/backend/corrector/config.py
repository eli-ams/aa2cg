"""Defaults for the bundled fourth-stage table corrector.

The GUI worker replaces all project-specific paths and execution settings before
calling the optimizer. These local defaults also keep the module safe to inspect
or reuse without referring to another repository folder.
"""

from pathlib import Path


SOURCE_LAMMPS_DIR = Path("output/lammps")
WORK_DIR = Path("runs")
BEST_DIR = Path("best")
SIMULATION_DATA_FILE = "simulation_data.out"
# Empty preserves the source input exactly. GUI overrides are per system.
SIMULATION_PROTOCOL = {}
# Empty means keep everything. Applied only to this run's completed trial copies.
TRIAL_STORAGE = {}
# Current broad search remains available; the first trial is an untouched baseline.
SEARCH = {}
OPTIMIZATION_DIR = Path('optimization')
CONTINUE_RUN = False
ADD_EXPLORE = 0
ADD_REFINE = 20

# This is run inside each trial folder.
LAMMPS_COMMAND = ["mpiexec", "-np", "16", "lmp", "-in", "input.data"]

# Optuna settings.
STUDY_NAME = "property_correction"
N_TRIALS = 30
RANDOM_SEED = 12345

# Pair-table Gaussian correction:
# U_new(r) = U_old(r) + amplitude * exp(-0.5 * ((r - center) / sigma)^2)
# Negative amplitude deepens attraction; positive amplitude adds repulsion.
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
        "min": 0.7,
        "max": 1.3,
        "max_abs_drift": 0.3,
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

# Optional structural loss using OpenMSCG cgib distributions.
# The mapped AA reference is calculated once; every trial calculates only CG.
INCLUDE_AA_CG_IN_LOSS = False
AA_CG_LOSS_WEIGHT = 1.0
AA_CG_TOPOLOGY = Path("output/work/top.in")
AA_CG_REFERENCE_TRAJECTORY = Path("output/work/types_cg.lammpstrj")
AA_CG_TRIAL_TRAJECTORY = "cg.lammpstrj"
AA_CG_REFERENCE_DIR = Path("aa_cg_reference")
AA_CG_MIN = 0.5
AA_CG_MAX = 8.0
AA_CG_CUTOFF = 10.0
AA_CG_BINS = 200
