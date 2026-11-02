"""Validated, display-independent protocol for Workbench-generated trial inputs."""
from __future__ import annotations

import math
import re


DEFAULT_PROTOCOL = dict(production_steps=500_000, equilibration_steps=0,
                        timestep_fs=1.0, temperature_k=298.15, pressure_atm=1.0,
                        ramp_ps=0.0)
SMOKE_MARKER = '# OPENMSCG_SMOKE_TEST_END'


def normalize_protocol(values):
    if set(values) - (set(DEFAULT_PROTOCOL) | {'ramp_steps'}):
        raise ValueError('Unrecognized simulation protocol setting.')
    result = dict(DEFAULT_PROTOCOL, **values)
    if 'ramp_steps' in values and 'ramp_ps' not in values:
        result['ramp_ps'] = 0.
    for key in result:
        value = float(result[key])
        if not math.isfinite(value):
            raise ValueError('Simulation protocol values must be finite.')
        if key.endswith('_steps'):
            if value != int(value) or not 0 <= value <= 2_000_000_000:
                raise ValueError('Step counts must be whole numbers between 0 and 2,000,000,000.')
            result[key] = int(value)
        else:
            result[key] = value
    if result['production_steps'] < 200:
        raise ValueError('Use at least 200 production steps to record two property samples.')
    if result['timestep_fs'] <= 0 or result['temperature_k'] <= 0:
        raise ValueError('Timestep and temperature must be positive.')
    if not 0 <= result['ramp_ps'] <= 1_000_000:
        raise ValueError('Ramp-up duration must be between 0 and 1,000,000 ps.')
    if result.get('ramp_steps', 0) and result['ramp_ps']:
        raise ValueError('Specify ramp-up steps, not both steps and a legacy duration.')
    if 'ramp_steps' in result and result['ramp_steps'] == 0:
        result.pop('ramp_steps')
    if result.get('ramp_steps', 0) or result['ramp_ps']:
        ramp_stages(result)
    return result


def merge_protocol(source, overrides):
    """Replace legacy time-based ramps when explicit step settings are supplied."""
    settings = dict(source)
    if 'ramp_steps' in overrides:
        settings['ramp_ps'] = 0.
    elif 'ramp_ps' in overrides:
        settings.pop('ramp_steps', None)
    settings.update(overrides)
    return normalize_protocol(settings)


def ramp_stages(protocol):
    """Three approximately equal-duration NVT stages; read legacy ps protocols too."""
    if protocol.get('ramp_steps', 0):
        total = int(protocol['ramp_steps'])
        if total < 3:
            raise ValueError('Use at least 3 ramp-up steps, or switch ramp-up off.')
        # Allocate the exact requested total in a 4:2:1 ratio: dt/4, dt/2, dt.
        counts = [total * weight // 7 for weight in (4, 2, 1)]
        remainders = [total * weight % 7 for weight in (4, 2, 1)]
        for index in sorted(range(3), key=lambda i: (-remainders[i], i))[:total - sum(counts)]:
            counts[index] += 1
        if counts[2] == 0:
            counts[0] -= 1
            counts[2] = 1
        return [(steps, float(protocol['timestep_fs']) * fraction)
                for steps, fraction in zip(counts, (.25, .5, 1.))]
    duration_fs = float(protocol['ramp_ps']) * 1000. / 3.
    stages = []
    for fraction in (.25, .5, 1.):
        dt = float(protocol['timestep_fs']) * fraction
        steps = max(1, int(math.ceil(duration_fs / dt)))
        if steps > 2_000_000_000:
            raise ValueError('Ramp-up has too many steps; reduce its duration or increase timestep.')
        stages.append((steps, dt))
    return stages


def duration_text(steps, timestep_fs):
    ps = steps * timestep_fs / 1000.
    return '{:g} ns'.format(ps / 1000.) if ps >= 1000 else '{:g} ps'.format(ps)


def protocol_summary(values):
    p = normalize_protocol(values)
    summary = 'Equilibrate {} → sample {} · {:g} fs · {:g} K · {:g} atm'.format(
        duration_text(p['equilibration_steps'], p['timestep_fs']),
        duration_text(p['production_steps'], p['timestep_fs']),
        p['timestep_fs'], p['temperature_k'], p['pressure_atm'])
    if p.get('ramp_steps', 0) or p['ramp_ps']:
        elapsed_fs = sum(steps * dt for steps, dt in ramp_stages(p))
        summary = 'Ramp {} → '.format(duration_text(1, elapsed_fs)) + summary
    return summary


def run_lines(protocol, dump_attributes, *, temperature_damping=100.0,
              pressure_damping=1000.0, seed=1234, dump_every=250):
    """Original NPT protocol, with optional unscored warm-up before sampling."""
    p = normalize_protocol(protocol)
    npt = ('fix npt all npt temp {0} {0} {1} iso {2} {2} {3}'.format(
        p['temperature_k'], temperature_damping, p['pressure_atm'], pressure_damping))
    lines = ['', 'compute kepatom all ke/atom',
             'velocity all create {} {} rot no dist gaussian'.format(p['temperature_k'], seed)]
    if p.get('ramp_steps', 0) or p['ramp_ps']:
        marker = ('# Workbench ramp-up steps: {}'.format(p['ramp_steps']) if p.get('ramp_steps', 0)
                  else '# Workbench ramp-up ps: {!r}'.format(p['ramp_ps']))
        lines.extend([marker,
                      '# NVT, fixed box; quarter / half / full timestep; no sampling',
                      'fix ramp all nvt temp {0} {0} {1}'.format(p['temperature_k'], temperature_damping)])
        for steps, dt in ramp_stages(p):
            lines.extend(['timestep {}'.format(dt), 'run {} post no'.format(steps)])
        lines.extend(['unfix ramp', 'reset_timestep 0'])
    lines.extend([npt, 'timestep {}'.format(p['timestep_fs'])])
    if p['equilibration_steps']:
        lines.extend(['# Workbench equilibration: excluded from scoring and trajectory output',
                      'run {} post no'.format(p['equilibration_steps']),
                      'unfix npt', 'reset_timestep 0', npt])
    lines.extend([
        'fix simulation_data all ave/time 1 99 100 &\n'
        'v_time c_thermo_temp c_thermo_press v_sysvol v_sysdensity &\n'
        'v_cella v_cellb v_cellc v_etotal v_pe v_ke v_evdwl v_coulomb &\n'
        'v_sxx v_syy v_szz v_syz v_sxz v_sxy &\n'
        'v_xhi v_yhi v_zhi v_xlo v_ylo v_zlo v_surfacetension &\n'
        'file simulation_data.out',
        'dump trajectories all custom {} cg.lammpstrj {}'.format(dump_every, dump_attributes),
        'run {} post no'.format(p['production_steps']),
        'unfix simulation_data', 'undump trajectories', 'uncompute kepatom', 'unfix npt'])
    return lines


def _tokens(text):
    # Ignore formatting/comments, but do not accept unknown commands or variables.
    text = '\n'.join(line.partition('#')[0] for line in text.splitlines())
    tokens = text.replace('&', ' ').split()
    result = []
    for token in tokens:
        try:
            result.append(float(token))
        except ValueError:
            result.append(token)
    return result


def read_protocol(text):
    """Read only the generated literal protocol; custom scripts remain opt-in untouched."""
    error = ('This input is not a supported Workbench simulation protocol. '
             'Keep “Use source input” or regenerate the LAMMPS setup before customizing.')
    if text.count(SMOKE_MARKER) != 1:
        raise ValueError(error)
    prefix, tail = text.split(SMOKE_MARKER)
    units = re.findall(r'^\s*units\s+(\S+)\s*(?:#.*)?$', prefix, re.M)
    if units != ['real']:
        raise ValueError('Simulation protocol controls require LAMMPS units real.')
    clean = '\n'.join(line.partition('#')[0].strip() for line in tail.splitlines())
    def single(pattern):
        matches = re.findall(pattern, clean, re.M)
        if len(matches) != 1:
            raise ValueError(error)
        return matches[0]
    try:
        legacy_ramp = re.findall(r'^# Workbench ramp-up ps: (\S+)$', tail, re.M)
        step_ramp = re.findall(r'^# Workbench ramp-up steps: (\S+)$', tail, re.M)
        ramp = legacy_ramp or step_ramp
        if len(legacy_ramp) + len(step_ramp) > 1:
            raise ValueError(error)
        ramp_value = float(ramp[0]) if ramp else 0.
        if ramp and (not math.isfinite(ramp_value) or ramp_value <= 0):
            raise ValueError(error)
        timesteps = re.findall(r'^timestep\s+(\S+)$', clean, re.M)
        if len(timesteps) != (4 if ramp else 1):
            raise ValueError(error)
        timestep = float(timesteps[-1])
        temperature, seed = single(r'^velocity all create (\S+) (\d+) rot no dist gaussian$')
        npts = re.findall(r'^fix npt all npt temp (\S+) (\S+) (\S+) iso (\S+) (\S+) (\S+)$', clean, re.M)
        runs = re.findall(r'^run (\d+) post no$', clean, re.M)
        protocol_runs = runs[3:] if ramp else runs
        if len(protocol_runs) not in (1, 2) or len(npts) != len(protocol_runs):
            raise ValueError(error)
        t1, t2, tdamp, p1, p2, pdamp = map(float, npts[0])
        if (t1 != t2 or p1 != p2 or t1 != float(temperature) or
                not math.isfinite(tdamp) or not math.isfinite(pdamp) or tdamp <= 0 or pdamp <= 0):
            raise ValueError(error)
        every, attributes = single(r'^dump trajectories all custom (\d+) cg\.lammpstrj (.+)$')
        if int(every) <= 0 or int(seed) <= 0:
            raise ValueError(error)
        settings = normalize_protocol(dict(production_steps=int(protocol_runs[-1]),
            equilibration_steps=int(protocol_runs[0]) if len(protocol_runs) == 2 else 0,
            timestep_fs=timestep, temperature_k=t1, pressure_atm=p1,
            **({'ramp_steps': ramp_value} if step_ramp else {'ramp_ps': ramp_value})))
        extra = dict(temperature_damping=tdamp, pressure_damping=pdamp,
                     seed=int(seed), dump_every=int(every))
        if _tokens(tail) != _tokens('\n'.join(run_lines(settings, attributes, **extra))):
            raise ValueError(error)
        return settings, attributes, extra
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(error) from exc


def prepare_trial_input(text, overrides):
    if not overrides:
        return text  # Preserve all legacy/custom inputs exactly unless opted in.
    source, attributes, extra = read_protocol(text)
    settings = merge_protocol(source, overrides)
    prefix = text.split(SMOKE_MARKER)[0]
    return prefix + SMOKE_MARKER + '\n' + '\n'.join(run_lines(settings, attributes, **extra)) + '\n'


def validate_scoring_window(protocol, properties):
    samples = normalize_protocol(protocol)['production_steps'] // 100
    for item in properties:
        skip = float(item.get('skip_fraction', 0.))
        if not math.isfinite(skip) or not 0 <= skip < 1:
            raise ValueError('Discard initial % must be between 0 and less than 100.')
        if samples - int(samples * skip) < 2:
            raise ValueError('The scoring window for {} retains fewer than two samples. '
                             'Increase production steps or reduce discard %.'.format(item['column']))
