"""Remove Sphinx documentation build artifacts."""

import os
import shutil
import sys


def main():
    """
    Remove the documentation build directory and generated files.
    
    Cleans up the Sphinx build output in docs/_build/ and 
    generated documentation in docs/generated/
    """
    # Get the directory of this script (fuellib package)
    fuellib_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root (one level up from fuellib package)
    project_root = os.path.dirname(fuellib_dir)
    
    # Docs directory
    docs_dir = os.path.join(project_root, "docs")
    build_dir = os.path.join(docs_dir, "_build")
    generated_dir = os.path.join(docs_dir, "generated")
    
    # Remove build directory
    if os.path.exists(build_dir):
        try:
            shutil.rmtree(build_dir)
            print(f"Removed documentation build directory: {build_dir}")
        except Exception as e:
            print(f"Error removing build directory: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Build directory does not exist: {build_dir}")
    
    # Remove generated directory
    if os.path.exists(generated_dir):
        try:
            shutil.rmtree(generated_dir)
            print(f"Removed generated documentation directory: {generated_dir}")
        except Exception as e:
            print(f"Error removing generated directory: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Generated directory does not exist: {generated_dir}")


if __name__ == "__main__":
    main()
