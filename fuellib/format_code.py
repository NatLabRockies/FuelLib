"""Format all Python source code using Black."""

import os
import subprocess
import sys


def main():
    """Run Black formatter on all Python files in the repository."""
    # Get the directory of this script (fuellib package)
    fuellib_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the project root (one level up from fuellib package)
    project_root = os.path.dirname(fuellib_dir)

    try:
        result = subprocess.run(
            ["find", project_root, "-name", "*.py", "-print0"],
            capture_output=True,
            text=False,
            check=True,
        )

        # Use xargs to pass files to black
        process = subprocess.Popen(
            ["xargs", "-0", "black"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(input=result.stdout)

        if stdout:
            print(stdout.decode())
        if stderr:
            print(stderr.decode(), file=sys.stderr)

        sys.exit(process.returncode)
    except Exception as e:
        print(f"Error running black formatter: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
