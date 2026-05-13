"""
Build Sphinx documentation for FuelLib.

This script builds the HTML documentation using Sphinx, handling the proper
setup of paths and environment variables needed for autodoc to work correctly.
"""

import os
import sys
import subprocess


def main():
    """
    Build the FuelLib documentation.

    Changes to the docs directory and runs sphinx-build to generate HTML documentation.
    """
    # Get the directory of this script (fuellib package)
    fuellib_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the project root (one level up from fuellib package)
    project_root = os.path.dirname(fuellib_dir)

    # Docs directory
    docs_dir = os.path.join(project_root, "docs")

    # Ensure fuellib is in the Python path for autodoc
    if fuellib_dir not in sys.path:
        sys.path.insert(0, os.path.dirname(fuellib_dir))

    # Build command
    build_cmd = [
        "sphinx-build",
        "-M",
        "html",
        docs_dir,
        os.path.join(docs_dir, "_build"),
    ]

    print(f"Building documentation from {docs_dir}")
    print(f"Command: {' '.join(build_cmd)}")
    print()

    # Run sphinx-build
    result = subprocess.run(build_cmd)

    if result.returncode == 0:
        print()
        print("=" * 80)
        print("Documentation built successfully!")
        print(
            f"View the documentation at: {os.path.join(docs_dir, '_build', 'html', 'index.html')}"
        )
        print("=" * 80)
    else:
        print()
        print("Documentation build failed. Please check the errors above.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
