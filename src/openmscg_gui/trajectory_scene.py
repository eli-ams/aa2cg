"""Read-only projection of detected species and mapping rules into 3D scenes.

Preview bead centres are geometric centres, matching the 2D mapping canvas.
This module never writes mapping files or alters the scientific mapping weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from functools import lru_cache
from pathlib import Path
from threading import RLock
import numpy as np
from PySide6.QtGui import QColor

from .bead_colors import bead_color

from .trajectory_gpu import (
    SceneData, scene_from_lammps_data, _build_unwrap_levels, _unwrap_bonded_positions,
)
from .backend.openmscg_wrap.structure_detection import (
    read_lammps_structure, analyse_species, _custom_rule_atom_alignment, _component_alignment,
)


@dataclass
class Species:
    identity: str
    name: str
    molecules: list[np.ndarray]
    molecule_ids: list[int]


@dataclass
class MolecularModel:
    scene: SceneData
    species: list[Species]
    beads: list[np.ndarray]
    bead_labels: list[str]
    unwrap_levels: list
    mapping_error: str = ""
    bead_colors: list[str] = field(default_factory=list)


def bead_colour(label, saved_color=None):
    # Honor each rule's saved color, including same-label rules in other species.
    color = QColor(saved_color or bead_color(label))
    if not color.isValid():
        color = QColor(bead_color(label))
    return color.getRgbF()[:3]


def model_bead_colour(model, index):
    saved = model.bead_colors[index] if index < len(model.bead_colors) else None
    return bead_colour(model.bead_labels[index], saved)


_model_lock = RLock()


def load_model(path, summaries, mode, rules):
    path = Path(path)
    stat = path.stat()
    # Share immutable topology/mapping across the three AA views. Coordinate
    # arrays are copied by make_scene, never changed on the cached model.
    with _model_lock:
        return _cached_model(str(path), stat.st_size, stat.st_mtime_ns,
                             json.dumps(summaries, sort_keys=True), mode,
                             json.dumps(rules, sort_keys=True))


@lru_cache(maxsize=2)
def _cached_model(path, size, modified, summaries_json, mode, rules_json):
    summaries, rules = json.loads(summaries_json), json.loads(rules_json)
    scene = scene_from_lammps_data(path)
    structure = read_lammps_structure(path)
    groups = analyse_species(structure)
    names = {s['id']: s.get('name', s['id']) for s in summaries}
    id_index = {int(a): i for i, a in enumerate(scene.atom_ids)}
    species, beads, labels, colors = [], [], [], []
    mapping_error = ""
    for group in groups:
        species.append(Species(
            group.species_id, names.get(group.species_id, group.formula),
            [np.array([id_index[a] for a in c.atom_ids], dtype=int) for c in group.components],
            [c.molecule_id for c in group.components],
        ))
        representative = group.components[0]
        selected_rules = [r for r in rules if r.get('species_id') == group.species_id]
        try:
            templates = []
            if mode == 'manual_custom' and selected_rules:
                alignment = _custom_rule_atom_alignment(selected_rules, representative)
                for rule in selected_rules:
                    for match in rule.get('matches', []):
                        members = [alignment[int(a)] for a in match['atoms']]
                        if not members:
                            raise ValueError('A saved bead has no atoms.')
                        templates.append((rule['bead_label'], members, rule.get('color')))
            elif mode == 'united_atom':
                for atom in representative.mol.GetAtoms():
                    if atom.GetAtomicNum() != 1:
                        members = [atom.GetIdx()] + [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                        templates.append((atom.GetSymbol() + 'H' + str(len(members) - 1), members, None))
            assigned = [a for _, members, _ in templates for a in members]
            if len(set(assigned)) != len(assigned):
                raise ValueError('Saved bead rules overlap. Resolve the mapping before previewing beads.')
            for component in group.components:
                alignment = _component_alignment(representative, component, structure)
                for label, members, color in templates:
                    beads.append(np.array([id_index[component.atom_ids[alignment[a]]] for a in members], dtype=int))
                    labels.append(label)
                    colors.append(color or bead_color(label))
        except (ValueError, KeyError, IndexError) as exc:
            mapping_error = str(exc)
            beads, labels, colors = [], [], []
            break
    # A failed mapping must not hide any species from the AA view.
    if mapping_error:
        species = [Species(g.species_id, names.get(g.species_id, g.formula),
                   [np.array([id_index[a] for a in c.atom_ids], dtype=int) for c in g.components],
                   [c.molecule_id for c in g.components]) for g in groups]
    return MolecularModel(scene, species, beads, labels,
                          _build_unwrap_levels(len(scene.atom_data), scene.topology_bond_indices), mapping_error, colors)


def load_cg_model(path, aa_model):
    scene = scene_from_lammps_data(path, coarse_grained=True)
    if len(scene.atom_data) != len(aa_model.beads) or not np.array_equal(scene.atom_ids, np.arange(1, len(aa_model.beads) + 1)):
        raise ValueError('CG topology does not match the current bead mapping. Regenerate CG outputs first.')
    # Backend emits beads in species / molecule / rule order, exactly as load_model.
    owner = {}
    for si, species in enumerate(aa_model.species):
        for mi, molecule in enumerate(species.molecules):
            for atom in molecule:
                owner[int(atom)] = (si, mi)
    grouped = [[[] for _ in s.molecules] for s in aa_model.species]
    for i, bead in enumerate(aa_model.beads):
        si, mi = owner[int(bead[0])]
        if any(owner[int(a)] != (si, mi) for a in bead):
            raise ValueError('A bead spans multiple molecules.')
        grouped[si][mi].append(i)
        scene.atom_data[i, 3] = 1.0
        scene.atom_data[i, 4:7] = model_bead_colour(aa_model, i)
    species = [Species(s.identity, s.name, [np.array(m, dtype=int) for m in grouped[i]], s.molecule_ids)
               for i, s in enumerate(aa_model.species)]
    return MolecularModel(scene, species, [], [],
                          _build_unwrap_levels(len(scene.atom_data), scene.topology_bond_indices))


def ordered_positions(model, frame, *, pbc=True):
    if frame is None:
        positions = model.scene.atom_data[:, :3].copy()
        cell, periodic, wrapped = model.scene.cell_matrix, model.scene.periodic, model.scene.positions_wrapped
    else:
        order = np.argsort(frame.atom_ids)
        if not np.array_equal(frame.atom_ids[order], model.scene.atom_ids):
            raise ValueError('Trajectory atom IDs do not match this structure (missing or duplicate IDs).')
        positions = frame.positions[order]
        cell, periodic, wrapped = frame.cell_matrix, frame.periodic, frame.positions_wrapped
    if pbc and wrapped and cell is not None and periodic is not None:
        positions = _unwrap_bonded_positions(positions, model.unwrap_levels, cell, periodic)
    return positions


def make_scene(model, frame=None, *, scope=0, species_index=0, molecule_index=0,
               representation='aa', dim=False, follow=True, pbc=True, show_box=False):
    from .trajectory_gpu import box_line_vertices
    # Isolated molecules are always whole; this never changes cached source data.
    pbc = bool(pbc or scope == 2)
    positions = ordered_positions(model, frame, pbc=pbc)
    source = frame if frame is not None else model.scene
    cell, periodic, wrapped = source.cell_matrix, source.periodic, source.positions_wrapped
    split_display = not pbc and wrapped and cell is not None and periodic is not None
    inverse_cell = np.linalg.inv(cell.astype('f8')) if split_display else None
    # Bead geometry must never be averaged across opposite sides of the cell.
    whole_positions = (ordered_positions(model, frame) if representation != 'aa' and not pbc
                       else positions)
    count = len(positions)
    selected = np.ones(count, dtype=bool)
    if scope and model.species:
        species = model.species[min(species_index, len(model.species) - 1)]
        members = species.molecules if scope == 1 else [species.molecules[min(molecule_index, len(species.molecules) - 1)]]
        selected[:] = False
        if members:
            selected[np.concatenate(members)] = True
    if not np.any(selected):
        raise ValueError('No atoms belong to this selection.')
    centre = positions[selected].mean(axis=0) if follow and scope else np.zeros(3)
    positions = positions - centre
    whole_positions = whole_positions - centre
    shown = np.ones(count, dtype=bool) if dim else selected.copy()
    atoms = model.scene.atom_data.copy()
    atoms[:, :3] = positions
    atoms[~selected, 4:7] = (0.12, 0.15, 0.19)
    atom_to_bead = np.full(count, -1, dtype=int)
    mapped, bead_rows = [], []
    if representation != 'aa':
        for bead_index, members in enumerate(model.beads):
            if not np.all(shown[members]):
                continue
            xyz = whole_positions[members].mean(axis=0)
            radius = max(0.8, float((np.linalg.norm(whole_positions[members] - xyz, axis=1) + atoms[members, 3]).max()))
            if split_display:
                fractional = (xyz + centre) @ inverse_cell.T
                fractional[periodic] -= np.floor(fractional[periodic] + .5)
                xyz = fractional @ cell.T - centre
            colour = model_bead_colour(model, bead_index) if np.any(selected[members]) else (0.12, 0.15, 0.19)
            atom_to_bead[members] = len(bead_rows)
            bead_rows.append([*xyz, radius, *colour])
            mapped.extend(members)
    if representation == 'cg':
        shown[mapped] = False  # Keep genuinely unmapped atoms visible.
    aa_indices = np.flatnonzero(shown)
    rows = atoms[aa_indices]
    opaque = len(rows)
    if bead_rows:
        rows = np.concatenate([rows, np.array(bead_rows, dtype='f4')])
    bonds = []
    if representation != 'cg':
        for a, b in model.scene.topology_bond_indices:
            if shown[a] and shown[b]:
                bonds.append([*positions[a], *positions[b], 0.11, *atoms[a, 4:7], *atoms[b, 4:7], 0])
    else:
        # Collapse atom bonds onto beads; retain connections to unmapped atoms.
        lookup = {int(a): i for i, a in enumerate(aa_indices)}
        edges = set()
        for a, b in model.scene.topology_bond_indices:
            ia = opaque + atom_to_bead[a] if atom_to_bead[a] >= 0 else lookup.get(int(a))
            ib = opaque + atom_to_bead[b] if atom_to_bead[b] >= 0 else lookup.get(int(b))
            if ia is not None and ib is not None and ia != ib:
                edges.add(tuple(sorted((ia, ib))))
        for a, b in edges:
            bonds.append([*rows[a, :3], *rows[b, :3], 0.15, *rows[a, 4:7], *rows[b, 4:7], 0])
    bond_data = np.asarray(bonds, dtype='f4').reshape(-1, 14)
    if split_display and len(bond_data):
        # Do not draw misleading box-spanning bonds in the as-stored display.
        fractional = (bond_data[:, 3:6] - bond_data[:, :3]) @ inverse_cell.T
        bond_data = bond_data[~np.any(np.abs(fractional[:, periodic]) > .5, axis=1)]
    box_vertices = box_line_vertices(cell) - centre if show_box else np.empty((0, 3), dtype='f4')
    focus = positions[selected]
    focus_centre = focus.mean(axis=0)
    radius = max(1., float(np.linalg.norm(focus - focus_centre, axis=1).max()) + 1.5)
    if scope == 0 and len(box_vertices):
        radius = max(radius, float(np.linalg.norm(box_vertices - focus_centre, axis=1).max()))
    scene = SceneData(np.asarray(rows, dtype='f4'), bond_data,
                      len(bond_data), focus_centre.astype('f4'), radius, model.scene.label,
                      cell_matrix=cell, periodic=periodic, box_vertices=np.asarray(box_vertices, dtype='f4'))
    return scene, opaque if representation == 'overlay' and bead_rows else None
