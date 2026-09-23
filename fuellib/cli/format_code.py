"""Format, lint, and type-check all Python source code, mirroring CI's Formatting job."""

import os
import subprocess
import sys


def main():
    """Run ruff format, ruff check --fix, and ty check on the repository."""
    # Get the directory of this script (fuellib/cli)
    cli_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the fuellib package directory (one level up from cli)
    fuellib_dir = os.path.dirname(cli_dir)

    # Get the project root (one level up from fuellib package)
    project_root = os.path.dirname(fuellib_dir)

    # Mirrors the CI "Formatting" job (ruff format --check, ruff check, ty check),
    # but applies fixes locally instead of just checking.
    commands = [
        [sys.executable, "-m", "ruff", "format", project_root],
        [sys.executable, "-m", "ruff", "check", project_root, "--fix"],
        [sys.executable, "-m", "ty", "check", project_root],
    ]

    exit_code = 0
    for command in commands:
        try:
            result = subprocess.run(command, check=False)
        except OSError as e:
            print(f"Error running {command[2]}: {e}", file=sys.stderr)
            exit_code = 1
            continue
        if result.returncode != 0:
            exit_code = result.returncode

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
