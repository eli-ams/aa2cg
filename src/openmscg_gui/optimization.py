from __future__ import annotations

from copy import deepcopy


# These are the thermodynamic columns written by lammps_generation.py to
# simulation_data.out. Time and box-bound bookkeeping columns are intentionally
# excluded from the optimization objective picker.
PROPERTY_CATALOG = [
    {"column": "v_sysdensity", "label": "Density", "tolerance": 0.05, "min": 0.7, "max": 1.3, "max_abs_drift": 0.3},
    {"column": "c_thermo_temp", "label": "Temperature", "tolerance": 5.0},
    {"column": "c_thermo_press", "label": "Pressure", "tolerance": 50.0},
    {"column": "v_sysvol", "label": "System volume", "tolerance": 100.0},
    {"column": "v_cella", "label": "Cell length A", "tolerance": 1.0},
    {"column": "v_cellb", "label": "Cell length B", "tolerance": 1.0},
    {"column": "v_cellc", "label": "Cell length C", "tolerance": 1.0},
    {"column": "v_etotal", "label": "Total energy", "tolerance": 100.0},
    {"column": "v_pe", "label": "Potential energy", "tolerance": 100.0},
    {"column": "v_ke", "label": "Kinetic energy", "tolerance": 100.0},
    {"column": "v_evdwl", "label": "van der Waals energy", "tolerance": 100.0},
    {"column": "v_coulomb", "label": "Coulomb energy", "tolerance": 100.0},
    {"column": "v_sxx", "label": "Stress XX", "tolerance": 50.0},
    {"column": "v_syy", "label": "Stress YY", "tolerance": 50.0},
    {"column": "v_szz", "label": "Stress ZZ", "tolerance": 50.0},
    {"column": "v_syz", "label": "Stress YZ", "tolerance": 50.0},
    {"column": "v_sxz", "label": "Stress XZ", "tolerance": 50.0},
    {"column": "v_sxy", "label": "Stress XY", "tolerance": 50.0},
    {"column": "v_surfacetension", "label": "Surface tension", "tolerance": 10.0},
]


def default_optimization_properties() -> list[dict]:
    return []


def property_definition(column: str) -> dict:
    for definition in PROPERTY_CATALOG:
        if definition["column"] == column:
            return definition
    raise KeyError(f"Unknown simulation property: {column}")


def property_settings(
    column: str,
    *,
    target: float | None = None,
    weight: float = 1.0,
    tolerance: float | None = None,
) -> dict:
    definition = deepcopy(property_definition(column))
    definition.pop("label", None)
    default_tolerance = float(definition.pop("tolerance"))
    scale = float(tolerance if tolerance is not None else default_tolerance)
    settings = {
        "column": column,
        "target": target,
        "weight": float(weight),
        "tail_scale": scale,
        "drift_scale": scale,
        "std_scale": scale,
        "slope_scale": scale,
        "skip_fraction": 0.0,
        "min": definition.get("min"),
        "max": definition.get("max"),
        "max_abs_drift": definition.get("max_abs_drift"),
        "hard_penalty": 1.0e6,
    }
    return settings

