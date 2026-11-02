"""Per-interaction overrides, separate from the generated automatic fit plan."""
from copy import deepcopy
import csv
import math
from pathlib import Path


MAX_SPLINE_INTERVALS = 10000


def spline_overrides(override):
    values = {}
    if 'resolution' in override:
        try:
            spacing = float(override['resolution'])
        except (TypeError, ValueError):
            raise ValueError('Spline spacing must be numeric.')
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError('Spline spacing must be finite and positive.')
        values['resolution'] = spacing
    if 'order' in override:
        try:
            order = float(override['order'])
        except (TypeError, ValueError):
            raise ValueError('Spline order must be an integer.')
        if not math.isfinite(order) or not order.is_integer() or not 2 <= order <= 6:
            raise ValueError('Choose an integer spline order between 2 and 6.')
        values['order'] = int(order)
    return values


def validate_override(entry, override):
    if entry.get('kind') not in ('Pair', 'Bond', 'Angle', 'Dihedral'):
        raise ValueError('Manual fit settings currently support pairs, bonds and angles.')
    if not isinstance(override, dict):
        raise ValueError('Manual fit settings must contain lower/upper bounds and spacing.')
    try:
        values = {key: float(override[key]) for key in ('min', 'max', 'resolution')}
    except (KeyError, TypeError, ValueError):
        raise ValueError('Enter numeric fitting bounds and spline spacing.')
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Fitting bounds and spacing must be finite.')
    lo, hi, spacing = (values[key] for key in ('min', 'max', 'resolution'))
    dihedral = entry['kind'] == 'Dihedral'
    if dihedral and (lo != entry['fit_params']['min'] or hi != entry['fit_params']['max']):
        raise ValueError('Dihedral bounds must retain the original periodic domain.')
    if (lo < 0 and not dihedral) or hi <= lo:
        raise ValueError('The lower bound must be nonnegative and below the upper bound.')
    if entry['kind'] == 'Angle' and hi > 180:
        raise ValueError('Angle bounds must stay between 0 and 180 degrees.')
    if entry['kind'] == 'Pair':
        try:
            cutoff = float(entry['fit_params']['cutoff'])
        except (KeyError, TypeError, ValueError):
            raise ValueError('The generated plan has no valid pair cutoff; rebuild the mapped model.')
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError('The generated plan has no valid pair cutoff; rebuild the mapped model.')
        if hi > cutoff:
            raise ValueError('The upper bound cannot exceed the pair cutoff ({:g}).'.format(cutoff))
    if spacing <= 0 or spacing > hi - lo:
        raise ValueError('Spline spacing must be positive and no wider than the fitting interval.')
    if (hi - lo) / spacing > MAX_SPLINE_INTERVALS:
        raise ValueError('Increase spacing: at most 10,000 spline intervals are allowed.')
    if 'order' in override:
        values['order'] = spline_overrides(override)['order']
    if 'spacing_mode' in override:
        if override['spacing_mode'] not in ('default', 'custom'):
            raise ValueError('Invalid spline spacing mode.')
        values['spacing_mode'] = override['spacing_mode']
    return values


def automatic_defaults(entry):
    """Old plans omit provenance; leave their bounds intact until explicitly edited."""
    config = entry.get('auto_fit_settings', {})
    if not isinstance(config, dict):
        raise ValueError('Invalid original automatic settings in the fit plan.')
    method = str(config.get('mode', 'quantile')).lower()
    method = {'by_threshold': 'threshold', 'by_trail': 'threshold',
              'by_quantile': 'quantile'}.get(method, method)
    return {
        'mode': 'automatic', 'method': method if method in ('quantile', 'threshold') else 'quantile',
        'lower_quantile': config.get('lower_quantile', .005),
        'upper_quantile': config.get('upper_quantile', .995),
        'histogram_threshold_percent': config.get('histogram_threshold_percent', 5.),
    }


def normalize_override(entry, override):
    if not isinstance(override, dict):
        raise ValueError('Invalid fitting settings.')
    mode = override.get('mode', 'manual')  # Existing bounds-only overrides stay manual.
    if mode == 'manual':
        if entry.get('kind') == 'Dihedral':
            raise ValueError('Dihedral bounds retain the original periodic domain.')
        return validate_override(entry, override)
    if mode != 'automatic' or entry.get('kind') not in ('Pair', 'Bond', 'Angle', 'Dihedral'):
        raise ValueError('Unsupported automatic fitting settings.')
    original = override.get('original_bounds', False)
    if not isinstance(original, bool):
        raise ValueError('Invalid original-bounds flag.')
    if entry['kind'] == 'Dihedral' and not original:
        raise ValueError('Dihedral bounds retain the original periodic domain.')
    settings = automatic_defaults(entry)
    settings.update(override)
    if settings['method'] not in ('quantile', 'threshold'):
        raise ValueError('Choose Trim tails or Relative peak height.')
    try:
        values = {key: float(settings[key]) for key in
                  ('lower_quantile', 'upper_quantile', 'histogram_threshold_percent')}
    except (TypeError, ValueError):
        raise ValueError('Automatic limits must be numeric.')
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Automatic limits must be finite.')
    if not 0 <= values['lower_quantile'] < values['upper_quantile'] <= 1:
        raise ValueError('Tail percentiles must satisfy 0 ≤ lower < upper ≤ 100%.')
    if not 0 <= values['histogram_threshold_percent'] < 100:
        raise ValueError('Peak threshold must be at least 0% and below 100%.')
    result = dict(values, mode='automatic', method=settings['method'])
    result.update(spline_overrides(override))
    if original:
        result['original_bounds'] = True
    return result


def histogram_samples(root, entry):
    path = Path(root) / 'output' / 'histograms' / ('histogram_xy_' + entry['histogram_name'] + '.csv')
    if not path.is_file():
        path = Path(entry['histogram_csv'])
    pairs = []
    with path.open(encoding='utf-8') as stream:
        for row in csv.reader(stream):
            if len(row) < 2:
                continue
            try:
                pair = (float(row[0]), float(row[1]))
            except ValueError:
                continue
            if all(math.isfinite(value) for value in pair):
                pairs.append(pair)
    if not pairs:
        raise ValueError('No numeric histogram samples.')
    return sorted(pairs)


def resolve_override(entry, override, root=None, samples=None):
    settings = normalize_override(entry, override)
    if settings.get('mode') != 'automatic':
        return {key: value for key, value in settings.items() if key != 'spacing_mode'}
    if settings.get('original_bounds'):
        params = dict(entry['fit_params'])
        params.update(spline_overrides(settings))
        result = validate_override(entry, params)
        if 'order' not in settings:
            result.pop('order', None)
        return result
    from .backend.openmscg_wrap.fitting import infer_auto_fit_range, apply_auto_fit_range_to_params
    if samples is None:
        if root is None:
            raise ValueError('The histogram directory is required for automatic overrides.')
        samples = histogram_samples(root, entry)
    config = dict(settings, mode=settings['method'], enabled=True)
    bounds = infer_auto_fit_range(samples, config)
    try:
        bins = int(entry.get('auto_fit_settings', {}).get('bins', 200))
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Invalid automatic spline interval count in the fit plan.')
    if bins < 1:
        raise ValueError('The automatic spline interval count must be positive.')
    params = apply_auto_fit_range_to_params(entry['fit_params'], bounds, bins)
    params.update(spline_overrides(settings))
    result = validate_override(entry, params)
    if 'order' not in settings:
        result.pop('order', None)
    return result


def effective_fit_plan(plan, overrides, root=None):
    """Copy the plan; preserve every automatic parameter without an override."""
    result = deepcopy(plan)
    if not isinstance(overrides, dict):
        raise ValueError('Saved fitting overrides must be a dictionary.')
    for entry in result.get('entries', []):
        name = entry['name']
        if name in overrides:
            entry['fit_params'].update(resolve_override(entry, overrides[name], root=root))
    # Unmatched overrides remain saved, but never affect unrelated interactions.
    return result
