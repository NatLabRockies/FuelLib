import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ruff_command() -> list[str] | None:
    """Return the command used to invoke ruff, or None if ruff is unavailable."""
    probe = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", "ruff"]
    executable = shutil.which("ruff")
    return [executable] if executable else None


class StyleTestCase(unittest.TestCase):
    """Check that the code base follows the style rules configured in ruff.toml."""

    @classmethod
    def setUpClass(cls):
        cls.ruff = _ruff_command()
        if cls.ruff is None:
            raise unittest.SkipTest("ruff is not installed")

    def _run_ruff(self, *args):
        result = subprocess.run(
            [*self.ruff, "check", "--no-cache", "--output-format", "concise", *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        self.assertEqual(
            result.returncode, 0, f"ruff reported violations:\n{result.stdout}"
        )

    def test_pep8_naming(self):
        """Names follow PEP 8 everywhere except the Greenfield model.

        The exception for `greenfield.py` is configured in `ruff.toml`.
        """
        self._run_ruff("--select", "N", ".")

    def test_ruff_check(self):
        """All rules enabled in ruff.toml pass."""
        self._run_ruff(".")


if __name__ == "__main__":
    unittest.main()
