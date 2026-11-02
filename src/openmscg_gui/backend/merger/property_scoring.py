from __future__ import annotations

import math

import numpy as np


def score_properties(rows: list[dict[str, float]], properties: list[dict]) -> dict[str, object]:
    if not properties:
        raise ValueError("PROPERTIES must contain at least one property definition.")

    scored_properties = []
    total = 0.0
    for settings in properties:
        result = _score_one_property(rows, settings)
        scored_properties.append(result)
        total += result["weighted_score"]

    if not math.isfinite(total):
        total = 1.0e12
    return {
        "objective": "multi_property_drift",
        "score": float(total),
        "properties": scored_properties,
    }


def _score_one_property(rows: list[dict[str, float]], settings: dict) -> dict[str, object]:
    column = _required(settings, "column")
    if column not in rows[0]:
        raise ValueError(f"Column {column!r} not found in simulation_data.out.")

    skip_fraction = float(settings.get("skip_fraction", 0.0))
    start = int(len(rows) * skip_fraction)
    values = np.array([row[column] for row in rows[start:]], dtype=float)
    if len(values) == 0:
        raise ValueError(f"No values left for {column!r} after skip_fraction={skip_fraction}.")

    target = settings.get("target")
    reference = float(target) if target is not None else float(values[0])
    mean = float(np.mean(values))
    tail_mean = float(np.mean(values[len(values) // 2 :]))
    final = float(values[-1])
    std = float(np.std(values))
    drift = float(final - reference)
    tail_drift = float(tail_mean - reference)
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0.0

    tail_scale = _positive(settings, "tail_scale")
    drift_scale = _positive(settings, "drift_scale")
    std_scale = _positive(settings, "std_scale")
    slope_scale = float(settings.get("slope_scale", drift_scale))
    if slope_scale <= 0.0:
        raise ValueError(f"slope_scale must be positive for {column!r}.")

    score = (
        abs(tail_drift) / tail_scale
        + abs(drift) / drift_scale
        + std / std_scale
        + abs(slope) * len(values) / slope_scale
    )

    min_value = settings.get("min")
    max_value = settings.get("max")
    max_abs_drift = settings.get("max_abs_drift")
    hard_penalty = float(settings.get("hard_penalty", 1.0e6))
    guard_failed = False
    if min_value is not None and float(np.min(values)) < float(min_value):
        guard_failed = True
    if max_value is not None and float(np.max(values)) > float(max_value):
        guard_failed = True
    if max_abs_drift is not None and abs(drift) > float(max_abs_drift):
        guard_failed = True
    if guard_failed:
        score += hard_penalty

    if not math.isfinite(score):
        score = 1.0e12

    weight = float(settings.get("weight", 1.0))
    return {
        "column": column,
        "weight": weight,
        "score": float(score),
        "weighted_score": float(weight * score),
        "reference_value": reference,
        "mean_value": mean,
        "tail_mean_value": tail_mean,
        "final_value": final,
        "std_value": std,
        "drift": drift,
        "tail_drift": tail_drift,
        "slope_per_sample": slope,
        "guard_failed": guard_failed,
    }


def _required(settings: dict, key: str):
    if key not in settings:
        raise ValueError(f"Missing required property setting: {key}")
    return settings[key]


def _positive(settings: dict, key: str) -> float:
    value = float(_required(settings, key))
    if value <= 0.0:
        raise ValueError(f"{key} must be positive for {settings.get('column')!r}.")
    return value
