from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PublicationCheckerTests(unittest.TestCase):
    def test_current_tree_passes(self) -> None:
        if not (ROOT / ".git").exists():
            self.skipTest("publication tree check requires a Git checkout")
        completed = subprocess.run(
            [sys.executable, "scripts/check_publication.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
