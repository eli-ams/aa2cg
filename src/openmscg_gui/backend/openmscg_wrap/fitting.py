from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv


def interaction_key(kind: str, interaction: tuple[str, ...]) -> str:
    return f"{kind}-{'-'.join(interaction)}"


def params_for(config: dict, kind: str, interaction: tuple[str, ...]) -> dict:
    section = config.get(kind, {})
    joined = "-".join(interaction)
    if joined in section:
        return section[joined]
    if f"{kind}-{joined}" in section:
        return section[f"{kind}-{joined}"]
    if "all" not in section:
        raise KeyError(f"Missing [{kind}.all] parameters for {interaction}.")
    return section["all"]


def cgib_spec(params: dict, interaction: tuple[str, ...]) -> str:
    return f"{','.join(interaction)},min={params['min']},max={params['max']},bins={params['bins']}"


def cgfm_spec(params: dict, interaction: tuple[str, ...]) -> str:
    return (
        f"model={params['model']},type={':'.join(interaction)},"
        f"min={params['min']},max={params['max']},"
        f"resolution={params['resolution']},order={params['order']}"
    )


def cgdump_spec(kind: str, params: dict, interaction: tuple[str, ...]) -> str:
    return f"{kind}_{'-'.join(interaction)},{params['min']},{params['max']},{params['resolution']}"


def analyze_histogram_csvs(histogram_dir: Path, threshold_percent: float) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for path in histogram_dir.glob("histogram_xy_*.csv"):
        xs: list[float] = []
        ys: list[float] = []
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                xs.append(float(row[0]))
                ys.append(float(row[1]))
        if not ys:
            continue
        threshold = threshold_percent * max(ys) / 100.0
        trail = next((i for i, value in enumerate(ys) if value > threshold), None)
        lead = next((i for i, value in reversed(list(enumerate(ys))) if value > threshold), None)
        if trail is None or lead is None or xs[lead] <= xs[trail]:
            raise ValueError(f"Could not infer fitting range from {path}.")
        result[_remove_prefix(path.stem, "histogram_xy_")] = {
            "trail_x": xs[trail],
            "lead_x": xs[lead],
        }
    return result


def infer_range_from_histogram_data(data, threshold_percent: float) -> dict[str, float]:
    return infer_threshold_range_from_histogram_data(data, threshold_percent)


def infer_auto_fit_range(data, auto_fit: dict) -> dict[str, float] | None:
    if not bool(auto_fit.get("enabled", True)):
        return None
    mode = auto_fit.get("mode", "threshold").lower()
    aliases = {
        "by_threshold": "threshold",
        "by_trail": "threshold",
        "by_quantile": "quantile",
    }
    mode = aliases.get(mode, mode)
    if mode == "manual":
        return None
    if mode == "threshold":
        return infer_threshold_range_from_histogram_data(
            data,
            float(auto_fit.get("histogram_threshold_percent", 5.0)),
        )
    if mode == "quantile":
        return infer_quantile_range_from_histogram_data(
            data,
            float(auto_fit.get("lower_quantile", 0.005)),
            float(auto_fit.get("upper_quantile", 0.995)),
        )
    raise ValueError(f"Unsupported AUTO_FIT mode: {mode}")


def infer_threshold_range_from_histogram_data(data, threshold_percent: float) -> dict[str, float]:
    xs = [float(row[0]) for row in data]
    ys = [float(row[1]) for row in data]
    if not ys:
        raise ValueError("Cannot infer fit range from an empty histogram.")
    threshold = threshold_percent * max(ys) / 100.0
    trail = next((i for i, value in enumerate(ys) if value > threshold), None)
    lead = next((i for i, value in reversed(list(enumerate(ys))) if value > threshold), None)
    if trail is None or lead is None or xs[lead] <= xs[trail]:
        raise ValueError("Could not infer fitting range from histogram data.")
    return {"trail_x": xs[trail], "lead_x": xs[lead]}


def infer_quantile_range_from_histogram_data(
    data,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, float]:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
    xs = [float(row[0]) for row in data]
    weights = [max(float(row[1]), 0.0) for row in data]
    total = sum(weights)
    if not xs or total <= 0.0:
        raise ValueError("Cannot infer quantile range from an empty histogram.")

    lower_target = lower_quantile * total
    upper_target = upper_quantile * total
    cumulative = 0.0
    lower_x = xs[0]
    upper_x = xs[-1]
    lower_found = False
    for x, weight in zip(xs, weights):
        cumulative += weight
        if not lower_found and cumulative >= lower_target:
            lower_x = x
            lower_found = True
        if cumulative >= upper_target:
            upper_x = x
            break

    if upper_x <= lower_x:
        raise ValueError("Could not infer a valid quantile fitting range from histogram data.")
    return {"trail_x": lower_x, "lead_x": upper_x}


def apply_auto_fit_range_to_params(params: dict, histogram_range: dict, bins: int) -> dict:
    updated = deepcopy(params)
    updated["min"] = round(histogram_range["trail_x"], 3)
    updated["max"] = round(histogram_range["lead_x"], 3)
    updated["resolution"] = round((updated["max"] - updated["min"]) / bins, 8)
    return updated


def apply_auto_fit_ranges(fitting_config: dict, histogram_data: dict, bins: int) -> dict:
    updated = deepcopy(fitting_config)
    for name, values in histogram_data.items():
        kind, joined = name.split("-", 1)
        updated.setdefault(kind, {})
        updated[kind][joined] = deepcopy(updated[kind]["all"])
        updated[kind][joined]["min"] = round(values["trail_x"], 3)
        updated[kind][joined]["max"] = round(values["lead_x"], 3)
        width = updated[kind][joined]["max"] - updated[kind][joined]["min"]
        updated[kind][joined]["resolution"] = round(width / bins, 8)
    return updated


def _remove_prefix(value: str, prefix: str) -> str:
    if value.startswith(prefix):
        return value[len(prefix):]
    return value
