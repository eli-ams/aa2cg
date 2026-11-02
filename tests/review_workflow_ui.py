"""Manual GUI integration check using synthetic data; never launches a simulation.

Run with the GUI dependencies installed and src on PYTHONPATH. Saves screenshots
to the supplied directory (or a temporary directory). No user projects are read.
"""
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
import json
import sys
import time

from PySide6 import QtCore, QtWidgets
from openmscg_gui.app import STYLE, _light_palette
from openmscg_gui.main_window import MainWindow
from openmscg_gui.project import Project
from openmscg_gui.trajectory_gpu import configure_opengl
from openmscg_gui.backend.openmscg_wrap.structure_detection import read_lammps_structure, analyse_species
from test_trajectory_scene import WATER


def main():
    configure_opengl()
    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    app.setPalette(_light_palette())
    app.setStyleSheet(STYLE)
    screenshots = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(mkdtemp(prefix='cg-ui-pictures-'))
    screenshots.mkdir(parents=True, exist_ok=True)
    errors = []
    sys.excepthook = lambda kind, value, trace: errors.append(f'{kind.__name__}: {value}')
    with TemporaryDirectory(prefix='cg-ui-fixture-') as temporary:
        root = Path(temporary)
        structure = root / 'water.data'
        structure.write_text(WATER)
        atoms = read_lammps_structure(structure)
        dump = root / 'aa.dump'
        frames = []
        for frame in range(6):
            rows = [f'{a.atom_id} {a.position[0] + frame * .1} {a.position[1]} {a.position[2]}' for a in reversed(list(atoms.atoms.values()))]
            frames.append(f'ITEM: TIMESTEP\n{frame * 100}\nITEM: NUMBER OF ATOMS\n6\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id x y z\n' + '\n'.join(rows) + '\n')
        dump.write_text(''.join(frames))
        project = Project.create(root)
        system = project.add_system('Water preview')
        system.structure_file, system.trajectory = str(structure), str(dump)
        system.detected_species = [g.summary() for g in analyse_species(atoms)]
        system.custom_mapping_rules = [dict(species_id='species-1', species_name='Water', species_smiles='O',
                                            bead_label='W', matches=[dict(atoms=[0, 1, 2])])]
        system.stages['inspect'] = system.stages['cgmodel'] = 'complete'
        work = project.system_root(system) / 'output' / 'work'
        histograms = work.parent / 'histograms'
        work.mkdir(parents=True)
        histograms.mkdir()
        (histograms / 'histogram_xy_Pair-W-W.csv').write_text('x,count\n1,0\n2,1\n3,4\n4,7\n5,4\n6,1\n7,0\n')
        (work / 'fit_plan.json').write_text(json.dumps(dict(entries=[dict(name='Pair_W-W', kind='Pair',
            histogram_name='Pair-W-W', fit_params=dict(min=2, max=6))])))
        best = project.system_root(system) / 'optimization' / 'best'
        best.mkdir(parents=True)
        (best / 'structure.data').write_text('CG water\n\n2 atoms\n0 bonds\n1 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\nMasses\n\n1 18.015\n\nAtoms # full\n\n1 1 1 0 2 2 2\n2 2 1 0 5 5 5\n')
        cg_frame = 'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id x y z\n2 5 5 5\n1 2 2 2\n'
        (best / 'cg.lammpstrj').write_text(''.join(cg_frame.format(step=i * 500) for i in range(4)))
        system.stages['lammps'] = system.stages['optimize'] = 'complete'
        project.save()
        window = MainWindow()
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window._set_project(project)
        window.show()

        def wait_for(predicate, timeout=25):
            start = time.monotonic()
            while not predicate():
                app.processEvents()
                if errors:
                    raise AssertionError(errors)
                if time.monotonic() - start > timeout:
                    raise AssertionError('GUI operation timed out')
                time.sleep(.01)
            app.processEvents()

        wait_for(lambda: window.import_page.trajectory_view._frame is not None)
        for index, label in enumerate(['inspect', 'map', 'distributions', 'fit', 'tables', 'lammps', 'refine', 'validate']):
            window.tabs.setCurrentIndex(index)
            if index == 1:
                view = window.mapping_page.trajectory_view
                wait_for(lambda: view._frame is not None)
                view.instance.setValue(2)
                view.slider.setValue(4)
                wait_for(lambda: not view._busy and view._frame.index == 4 and not view._dirty)
                assert view.instance.value() == 2
                assert window.import_page.trajectory_view.slider.value() == 4
                assert view.renderer.overlay_start == 3
                assert view.renderer.ctx is not None, view.message.text()
                view.renderer.grabFramebuffer().save(str(screenshots / 'molecule-overlay.png'))
            app.processEvents()
            window.grab().save(str(screenshots / f'{label}.png'))
        window.validation_page.result_tabs.setCurrentIndex(2)
        view = window.validation_page.trajectory_view
        wait_for(lambda: view._frame is not None)
        assert view.slider.value() == 0  # Validation owns its relative timeline.
        assert view.instance.value() == 2
        assert window.mapping_page.trajectory_view.instance.value() == 2
        comparison = window.validation_page.comparison
        cg_view = window.validation_page.cg_view
        wait_for(lambda: cg_view._frame is not None)
        assert cg_view.instance.value() == 2
        assert cg_view.renderer.atom_count == 1
        assert cg_view.slider.value() == 0
        comparison.slider.setValue(comparison.slider.maximum())
        wait_for(lambda: not cg_view._busy and not view._busy
                 and cg_view._frame.index == len(cg_view._reader.frames) - 1
                 and view._frame.index == len(view._reader.frames) - 1)
        assert view.slider.value() == view.slider.maximum()
        assert cg_view.slider.value() == cg_view.slider.maximum()
        window.resize(1100, 700)
        window.tabs.setCurrentIndex(1)
        app.processEvents()
        window.grab().save(str(screenshots / 'map-compact.png'))
        assert window.width() == 1100, window.size()
        assert window.height() == 700, window.size()
        # Detail dialogs and saved mapping controls remain accessible.
        window.mapping_page.settings_dialog.show()
        app.processEvents()
        assert window.mapping_page.saved_rules_table.isVisible()
        window.mapping_page.settings_dialog.close()
        empty = Project.create(root / 'empty')
        window._set_project(empty)
        assert all(v._model is None and v.renderer.atom_count == 0 for v in window.trajectory_views)
        window.close()
        for view in window.trajectory_views:
            view._pool.waitForDone(5000)
        app.processEvents()
        if errors:
            raise AssertionError(errors)
    print('GUI integration passed; screenshots:', screenshots)


if __name__ == '__main__':
    main()
