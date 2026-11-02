import sys

from PySide6 import QtCore, QtWidgets
from .settings import SETTINGS
from .toolbar import SketchToolBar
from .widget import MoleculeCanvas


class MoleculeDrawerWindow(QtWidgets.QMainWindow):
    smilesAccepted = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Molecule drawer")
        self.canvas = MoleculeCanvas(self, editable=True)
        self.setCentralWidget(self.canvas)
        self.toolbar = SketchToolBar(self)
        self.addToolBar(self.toolbar)
        self.status = self.statusBar()
        self.toolbar.symSelected.connect(self.canvas.set_atom_symbol)
        self.toolbar.chainModeSelected.connect(self.canvas.set_chain_mode)
        self.toolbar.cyclohexaneModeSelected.connect(self.canvas.set_cyclohexane_mode)
        self.toolbar.benzeneModeSelected.connect(self.canvas.set_benzene_mode)
        self.toolbar.unitedAtomToggled.connect(self.canvas.set_united_atom_mode)
        self.toolbar.clearClicked.connect(self.clear_all)
        self.toolbar.copySmilesClicked.connect(self.copy_smiles)
        self.toolbar.pasteSmilesClicked.connect(self.paste_smiles)
        self.toolbar.savePngClicked.connect(self.save_png)
        self.toolbar.saveSvgClicked.connect(self.save_svg)
        self.toolbar.undoClicked.connect(self.canvas.undo)
        self.toolbar.snapGridClicked.connect(self.canvas.snap_components_to_grid)
        self.canvas.statusSummaryChanged.connect(self.update_status_summary)
        self.resize(int(SETTINGS["resolution_w"]), int(SETTINGS["resolution_h"]))

    def get_smiles(self):
        return self.canvas.get_smiles()

    def get_mol(self):
        return self.canvas.get_mol()

    def update_status_from_smiles(self, smiles):
        if not smiles:
            self.status.showMessage("Empty sketch")
            return
        try:
            mol = self.canvas.get_mol()
            from rdkit.Chem import rdMolDescriptors
            formula = rdMolDescriptors.CalcMolFormula(mol)
            mass = rdMolDescriptors.CalcExactMolWt(mol)
            self.status.showMessage(f"{smiles}    |    {formula}    |    {mass:.4f} g/mol")
        except Exception as exc:
            self.status.showMessage(str(exc))

    def update_status_summary(self, smiles, formula, mass):
        if not smiles:
            self.status.showMessage("Empty sketch")
            return

        self.status.showMessage(f"{smiles}    |    {formula}    |    {mass:.4f} g/mol")

    def copy_smiles(self):
        smiles = self.canvas.get_smiles()
        if not smiles:
            self.status.showMessage("Empty sketch")
            return
        QtWidgets.QApplication.clipboard().setText(smiles)
        self.status.showMessage("SMILES copied: " + smiles)

    def paste_smiles(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Paste SMILES", "Enter SMILES string:")
        if not ok or not text.strip():
            return
        try:
            self.canvas.append_smiles(text)
            self.status.showMessage("SMILES added")
        except Exception as exc:
            self.status.showMessage(str(exc))

    def save_png(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PNG", "sketch.png", "PNG files (*.png)")
        if not path:
            return
        try:
            self.canvas.save_png(path)
            self.status.showMessage("Saved: " + path)
        except Exception as exc:
            self.status.showMessage(str(exc))

    def save_svg(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save SVG", "sketch.svg", "SVG files (*.svg)")
        if not path:
            return
        try:
            self.canvas.save_svg(path)
            self.status.showMessage("Saved: " + path)
        except Exception as exc:
            self.status.showMessage(str(exc))

    def clean_structure(self):
        try:
            self.canvas.clean_structure()
            self.status.showMessage("Structure cleaned")
        except Exception as exc:
            self.status.showMessage(str(exc))

    def clear_all(self):
        self.canvas.clear_all()
        self.status.showMessage("Cleared")

    def accept_molecule(self):
        smiles = self.canvas.get_smiles()
        if not smiles:
            self.status.showMessage("Empty sketch")
            return
        self.smilesAccepted.emit(smiles)
        self.status.showMessage("Molecule selected: " + smiles)


def main():
    app = QtWidgets.QApplication(sys.argv)
    font = app.font()
    font.setPointSize(12)
    app.setFont(font)
    window = MoleculeDrawerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
