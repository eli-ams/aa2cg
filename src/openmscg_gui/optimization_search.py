"""Small, explicit search steps; no GUI or Optuna import required."""
from __future__ import annotations

from copy import deepcopy
import math


DEFAULT_SEARCH = dict(mode='explore_refine', baseline_first=True, step_percent=10.0,
                      explore_trials=30, refine_trials=20)


def normalize_search(values=None):
    result = dict(DEFAULT_SEARCH, **(values or {}))
    if set(result) != set(DEFAULT_SEARCH) or result['mode'] not in ('current', 'incremental', 'explore_refine'):
        raise ValueError('Unknown Gaussian search setting.')
    if not isinstance(result['baseline_first'], bool):
        raise ValueError('Baseline selection must be on or off.')
    try:
        step = float(result['step_percent'])
    except (ValueError, TypeError):
        raise ValueError('Incremental step size must be between 0 and 100%.')
    if not math.isfinite(step) or not 0 < step <= 100:
        raise ValueError('Incremental step size must be greater than 0 and at most 100%.')
    result['step_percent'] = step
    for key in ('explore_trials', 'refine_trials'):
        try:
            count = float(result[key])
        except (ValueError, TypeError):
            raise ValueError('Trial counts must be whole numbers from 0 to 100,000.')
        if not math.isfinite(count) or count != int(count) or not 0 <= count <= 100_000:
            raise ValueError('Trial counts must be whole numbers from 0 to 100,000.')
        result[key] = int(count)
    if result['mode'] == 'explore_refine' and result['explore_trials'] < 1:
        raise ValueError('A new Explore → Refine study needs at least one Explore trial.')
    return result


def valid_score(data):
    """Use the same eligibility rule for the search reference and retained best."""
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    try:
        if (data.get('failed', False) or not finite(data) or
                any(p.get('guard_failed', False) for p in data.get('properties', []))):
            return None
        score = float(data['score'])
        return score if math.isfinite(score) else None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


class IncrementalSearch:
    """One coordinate per proposal, always measured from the best VALID vector.

    Bounds passed to Optuna are fixed [-1, 1] relative steps for the whole study.
    Actual Gaussian values and the reference trial are persisted separately.
    No corrections are stacked onto already-corrected tables.
    """
    def __init__(self, initial, bounds, step_percent=10., baseline_first=True):
        self.reference = deepcopy(initial)
        self.bounds = dict(bounds)
        self.step = float(step_percent) / 100.
        self.baseline_first = baseline_first
        self.best_score = None
        self.best_number = None
        self.proposals = 0

    def propose(self, trial):
        names = sorted(self.reference)
        index = self.proposals
        self.proposals += 1
        name = names[max(0, index) % len(names)]
        field = ('amplitude', 'center', 'sigma')[(max(0, index) // len(names)) % 3]
        if self.reference[name]['amplitude'] == 0:
            field = 'amplitude'  # Moving an inactive Gaussian's center/width does nothing.
        low, high = self.bounds[field]
        suggest = getattr(trial, 'suggest_float', None) or trial.suggest_uniform
        unit = suggest(name.replace('-', '_').replace('.', '_') + '_' + field + '_step', -1., 1.)
        proposal = deepcopy(self.reference)
        value = self.reference[name][field] + unit * self.step * (high - low)
        proposal[name][field] = max(low, min(high, value))
        return proposal, dict(reference_trial=self.best_number,
                             changed_interaction=name, changed_parameter=field)

    def accept(self, number, settings, score):
        if score is not None and math.isfinite(score) and (self.best_score is None or score < self.best_score):
            self.best_number, self.best_score = number, score
            self.reference = deepcopy(settings)
            return True
        return False
