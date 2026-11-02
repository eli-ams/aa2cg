"""Bounded log inspection and human-readable trial outcomes (never edits tables)."""
from __future__ import annotations

import re
import json

from .optimization_search import valid_score


def log_tail(path, limit=65536):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            text = stream.read(limit).decode('utf-8', errors='replace')
        return ('[Last {} bytes]\n'.format(limit) if size > limit else '') + text
    except OSError as exc:
        return 'Log unavailable: {}'.format(exc)


def lammps_failure(log_path, structure_path, return_code):
    tail = log_tail(log_path)
    messages = [line.strip() for line in tail.splitlines()
                if line.lstrip().startswith('ERROR')]
    message = messages[0] if messages else 'LAMMPS exited with code {}; see its log.'.format(return_code)
    category = 'lammps_error'
    interaction = None
    if 'Pair distance < table inner cutoff' in message:
        category = 'pair_inner_cutoff'
    elif 'Pair distance > table outer cutoff' in message:
        category = 'pair_outer_cutoff'
    match = re.search(r'ijtype\s+(\d+)\s+(\d+)', message)
    if match:
        labels, inside = {}, False
        try:
            with structure_path.open(encoding='utf-8') as stream:
                for line in stream:
                    if line.strip() == 'Masses':
                        inside = True
                    elif inside and line.strip():
                        parts = line.split()
                        if not parts[0].isdigit():
                            break
                        labels[int(parts[0])] = (line.partition('#')[2].strip().split() or [parts[0]])[0]
        except (OSError, ValueError):
            pass
        first, second = map(int, match.groups())
        interaction = '{}–{}'.format(labels.get(first, str(first)), labels.get(second, str(second)))
    reason = message
    if interaction:
        reason = '{} pair: {}'.format(interaction, message)
    return dict(failed=True, score=1.e12, failure_category=category,
                failure_reason=reason, lammps_message=message, interaction=interaction,
                return_code=return_code)


def outcome_reason(data):
    if not isinstance(data, dict):
        return 'Incomplete score.'
    if data.get('failure_reason'):
        return str(data['failure_reason'])
    if data.get('failed'):
        return 'LAMMPS or scoring failed; see the trial log.'
    properties = data.get('properties') or []
    if not isinstance(properties, list):
        return 'Incomplete property score.'
    guards = [str(p.get('column', 'property')) for p in properties
              if isinstance(p, dict) and p.get('guard_failed')]
    if guards:
        return 'Property limits exceeded: ' + ', '.join(guards)
    if valid_score(data) is None:
        return 'Incomplete or non-finite score.'
    return 'Passed property limits.'


def load_trial_results(root):
    """Read small records only, including cleaned trials; never scan trajectories."""
    paths = [root / 'best']
    paths.extend(sorted((root / 'runs').glob('trial_*')))
    paths.extend(sorted((root / 'trial_records').glob('run_*/trial_*')))
    paths.extend(sorted((root / 'studies').glob('study_*/best/trial_*')))
    paths.extend(sorted((root / 'studies').glob('study_*/runs/trial_*')))
    paths.extend(sorted((root / 'studies').glob('study_*/trial_records/run_*/trial_*')))
    results = {}
    def read(path):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}
    for directory in paths:
        score = read(directory / 'score.json')
        record = read(directory / 'trial_record.json')
        metadata = read(directory / 'correction.json')
        if not score and not record:
            continue
        number = record.get('trial', metadata.get('trial'))
        started = record.get('started', metadata.get('created', ''))
        study_id = next((p.name for p in directory.parents if p.name.startswith('study_')), 'legacy')
        key = (study_id, str(started), str(number))
        if key in results:
            continue  # Prefer full best/run data over its identical archived record.
        attrs = record.get('user_attributes') or metadata.get('search_trial') or {}
        kind = ('Baseline' if attrs.get('baseline') else
                ('Refine' if attrs.get('phase') == 'refine' else
                 ('Explore' if attrs.get('phase') == 'explore' else
                  ('Incremental' if attrs.get('search_mode') == 'incremental' else 'Gaussian'))))
        try:
            value = float(score['score'])
        except (KeyError, TypeError, ValueError):
            value = None
        results[key] = dict(trial=number, started=started, kind=kind, score=value,
            study=study_id,
            outcome='Valid' if valid_score(score) is not None else 'Rejected',
            reason=outcome_reason(score), directory=directory,
            reference=attrs.get('reference_trial'), changed=attrs.get('changed_interaction'),
            parameter=attrs.get('changed_parameter'))
    return sorted(results.values(), key=lambda r: (str(r['started']), str(r['trial']).zfill(8)))
