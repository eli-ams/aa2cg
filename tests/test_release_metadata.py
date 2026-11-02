"""Release identity checks without loading Qt or executing scientific stages."""
from pathlib import Path
import re
import unittest

from openmscg_gui import APP_NAME, APP_TITLE, __version__


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_first_public_beta_identity(self):
        self.assertEqual(__version__, '0.1.0b1')
        self.assertEqual(APP_NAME, 'AA2CG')
        self.assertEqual(APP_TITLE, 'AA2CG 0.1.0 beta 1')

    def test_packaging_and_release_notes_agree(self):
        metadata = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
        project = metadata.split('[project]', 1)[1].split('\n[', 1)[0]
        self.assertEqual(re.search(r'^name = "([^"]+)"', project, re.M).group(1), 'aa2cg')
        self.assertEqual(re.search(r'^version = "([^"]+)"', project, re.M).group(1), __version__)
        for name in ('README.md', 'CHANGELOG.md'):
            text = (ROOT / name).read_text(encoding='utf-8')
            self.assertIn(__version__, text)
            self.assertIn('v0.1.0-beta.1', text)


if __name__ == '__main__':
    unittest.main()
