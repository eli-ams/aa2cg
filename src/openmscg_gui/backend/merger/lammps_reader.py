from __future__ import annotations

from pathlib import Path


SIMULATION_DATA_COLUMNS = [
    "TimeStep",
    "v_time",
    "c_thermo_temp",
    "c_thermo_press",
    "v_sysvol",
    "v_sysdensity",
    "v_cella",
    "v_cellb",
    "v_cellc",
    "v_etotal",
    "v_pe",
    "v_ke",
    "v_evdwl",
    "v_coulomb",
    "v_sxx",
    "v_syy",
    "v_szz",
    "v_syz",
    "v_sxz",
    "v_sxy",
    "v_xhi",
    "v_yhi",
    "v_zhi",
    "v_xlo",
    "v_ylo",
    "v_zlo",
    "v_surfacetension",
]


def read_simulation_data(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"LAMMPS simulation data file not found: {path}")

    rows: list[dict[str, float]] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != len(SIMULATION_DATA_COLUMNS):
                raise ValueError(
                    f"Expected {len(SIMULATION_DATA_COLUMNS)} columns in {path}, "
                    f"found {len(parts)}: {line}"
                )
            rows.append({key: float(value) for key, value in zip(SIMULATION_DATA_COLUMNS, parts)})

    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows

