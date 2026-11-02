"""Small, persisted settings for structural validation (not optimization)."""
from __future__ import annotations

import math


DEFAULT_SAMPLING = dict(maximum_frames=150, discard_percent=0.0)
MAXIMUM_FRAMES = 1000


def normalize_sampling(values=None):
    result = dict(DEFAULT_SAMPLING, **(values or {}))
    if set(result) != set(DEFAULT_SAMPLING):
        raise ValueError('Unknown validation sampling setting.')
    try:
        maximum = float(result['maximum_frames'])
        discard = float(result['discard_percent'])
    except (TypeError, ValueError):
        raise ValueError('Validation sampling requires numeric settings.')
    if not math.isfinite(maximum) or maximum != int(maximum) or not 2 <= maximum <= MAXIMUM_FRAMES:
        raise ValueError('Validation frame limit must be a whole number from 2 to 1,000.')
    if not math.isfinite(discard) or not 0 <= discard <= 95:
        raise ValueError('Initial discard must be between 0 and 95%.')
    return dict(maximum_frames=int(maximum), discard_percent=discard)


def sampled_indices(total, settings=None):
    """Deterministic, evenly spaced frame indices, including both window ends.

    The initial discard is rounded down to whole frames; even a one-frame
    trajectory remains usable. Frames are sampled independently for AA and CG.
    """
    settings = normalize_sampling(settings)
    if total < 1:
        raise ValueError('No complete trajectory frames are available.')
    first = int(total * settings['discard_percent'] / 100.0)
    count = min(total - first, settings['maximum_frames'])
    if count == 1:
        return [first]
    span = total - first - 1
    return [first + (index * span // (count - 1)) for index in range(count)]
