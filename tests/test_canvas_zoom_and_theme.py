"""Manual GUI regression cases for zoom geometry and control contrast."""
import os
from pathlib import Path
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtCore, QtGui, QtWidgets

from openmscg_gui.app import STYLE, _light_palette
from openmscg_gui.molecule_drawer import MoleculeCanvas
from openmscg_gui.molecule_drawer.widget import ATOM_CIRCLE_SCALE, ATOM_STYLES


class CanvasZoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_canvas(self):
        canvas = MoleculeCanvas()
        self.addCleanup(canvas.deleteLater)
        canvas.atom_circle_mode = True
        canvas.model.atoms = {1: (50., 50., 'C'), 2: (120., 50., 'I')}
        return canvas

    def test_circle_and_font_dimensions_follow_view_zoom_together(self):
        canvas = self.make_canvas()
        dimensions = []
        for zoom in (.5, 1., 2.):
            canvas.scale = zoom
            canvas._ensure_paint_caches()
            painter = Mock()
            canvas._draw_element_circle(painter, 50., 50., 'C')
            radius = painter.drawEllipse.call_args[0][1]
            self.assertEqual(radius, ATOM_CIRCLE_SCALE * ATOM_STYLES['C']['radius'])
            self.assertEqual(canvas._font.pointSizeF(), 14.)
            self.assertEqual(canvas._small_font.pointSizeF(), 9.)
            dimensions.append((radius * zoom, canvas._font.pointSizeF() * zoom))
        for small, large in zip(dimensions, dimensions[1:]):
            self.assertAlmostEqual(large[0], 2 * small[0])
            self.assertAlmostEqual(large[1], 2 * small[1])

    def test_bead_summary_circles_also_scale_in_model_space(self):
        canvas = self.make_canvas()
        canvas.model.atoms = {1: (50., 50., 'C')}
        canvas.bead_overlays = [dict(atom_ids=[1], color='#e68a36', label='A')]
        for zoom in (.5, 2.):
            canvas.scale = zoom
            canvas._ensure_paint_caches()
            painter = Mock()
            canvas._draw_bead_summary(painter, (0., 0., 200., 200.))
            self.assertEqual(painter.drawEllipse.call_args[0][1:], (20., 20.))

    def test_circle_edges_are_selectable_and_highlights_enclose_each_element(self):
        canvas = self.make_canvas()
        for zoom in (.5, 2.):
            canvas.scale = zoom
            # Both points lie outside the old nine-model-unit picking radius.
            for aid, offset in ((1, 12.), (2, 15.)):
                x, y, symbol = canvas.model.atoms[aid]
                self.assertEqual(canvas.nearest_atom(*canvas.to_view(x + offset, y)), aid)
                self.assertGreater(canvas._highlight_radius(aid, 9.), canvas._element_radius(symbol))
        canvas.atom_circle_mode = False
        self.assertIsNone(canvas.nearest_atom(*canvas.to_view(62., 50.)))
        self.assertFalse(canvas._editing_enabled)


class ThemeVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.original_style = self.app.styleSheet()
        self.original_palette = self.app.palette()
        self.app.setPalette(_light_palette())
        self.app.setStyleSheet(STYLE)

    def tearDown(self):
        self.app.setStyleSheet(self.original_style)
        self.app.setPalette(self.original_palette)

    def test_svg_assets_are_available_and_valid_for_source_or_installed_package(self):
        from openmscg_gui import app

        directory = Path(app.__file__).resolve().parent / 'assets' / 'theme'
        self.assertNotIn('@THEME_ASSETS@', STYLE)
        for name in ('check', 'partial', 'radio', 'chevron-down', 'chevron-up'):
            path = directory / (name + '.svg')
            self.assertEqual(ET.parse(str(path)).getroot().tag, '{http://www.w3.org/2000/svg}svg')
            self.assertIn(path.as_posix(), STYLE)
            self.assertFalse(QtGui.QIcon(str(path)).pixmap(QtCore.QSize(16, 16)).isNull())

    def checkbox_indicator(self, state, enabled=True):
        checkbox = QtWidgets.QCheckBox('Customize protocol')
        self.addCleanup(checkbox.deleteLater)
        checkbox.setTristate(True)
        checkbox.setCheckState(state)
        checkbox.setEnabled(enabled)
        checkbox.ensurePolished()
        checkbox.resize(checkbox.sizeHint())
        option = QtWidgets.QStyleOptionButton()
        checkbox.initStyleOption(option)
        rect = checkbox.style().subElementRect(QtWidgets.QStyle.SubElement.SE_CheckBoxIndicator,
                                              option, checkbox)
        image = QtGui.QImage(checkbox.size(), QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor('#ffffff'))
        checkbox.render(image)
        return image.copy(rect)

    def test_checkbox_states_remain_visually_distinct_on_light_backgrounds(self):
        unchecked = self.checkbox_indicator(QtCore.Qt.CheckState.Unchecked)
        checked = self.checkbox_indicator(QtCore.Qt.CheckState.Checked)
        partial = self.checkbox_indicator(QtCore.Qt.CheckState.PartiallyChecked)
        disabled = self.checkbox_indicator(QtCore.Qt.CheckState.Checked, enabled=False)
        self.assertNotEqual(unchecked, checked)
        self.assertNotEqual(checked, partial)
        self.assertNotEqual(checked, disabled)
        for image in (unchecked, checked, partial, disabled):
            self.assertFalse(image.isNull())
            self.assertTrue(any(image.pixelColor(x, y).lightness() < 180
                                for x in range(image.width()) for y in range(image.height())))

    def test_disabled_text_is_explicit_and_not_white_on_white(self):
        palette = _light_palette()
        for role in (QtGui.QPalette.ColorRole.WindowText, QtGui.QPalette.ColorRole.Text,
                     QtGui.QPalette.ColorRole.ButtonText):
            self.assertEqual(palette.color(QtGui.QPalette.ColorGroup.Disabled, role).name(), '#6f7d78')


if __name__ == '__main__':
    unittest.main()
