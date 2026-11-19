from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_release


class ReleaseCheckTests(unittest.TestCase):
    def test_ignored_results_are_excluded_but_candidate_sources_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            (root / ".gitignore").write_text("results/\n", encoding="utf-8")
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            (root / "results").mkdir()
            (root / "results" / "model.pt").write_bytes(b"local checkpoint")
            (root / "candidate.pt").write_bytes(b"accidental checkpoint")

            with patch.object(check_release, "PROJECT_ROOT", root):
                names = {path.relative_to(root).as_posix() for path in check_release.release_files()}
                self.assertIn("module.py", names)
                self.assertNotIn("results/model.pt", names)
                self.assertIn("candidate.pt", names)
                with self.assertRaisesRegex(SystemExit, "candidate.pt"):
                    check_release.check_large_or_blocked_files()


if __name__ == "__main__":
    unittest.main()
