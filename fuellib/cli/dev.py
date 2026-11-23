"""Developer commands: format, test, and build, clean, or view the documentation."""

import argparse
import subprocess
import sys
from pathlib import Path

from . import build_docs, clean_docs, format_code

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _test(args: argparse.Namespace) -> None:
    """Run the test suite with pytest from the repository root."""
    command = [sys.executable, "-m", "pytest", *args.pytest_args]
    sys.exit(subprocess.run(command, cwd=_REPO_ROOT, check=False).returncode)


def _view_docs(args: argparse.Namespace) -> None:
    """Open the built HTML documentation in a browser."""
    script = Path(__file__).with_name("view-docs.sh")
    sys.exit(subprocess.run(["bash", str(script)], check=False).returncode)


def add_parser(commands: "argparse._SubParsersAction") -> None:
    """Add the ``dev`` command and its subcommands.

    Args:
        commands: Subparsers action of the ``fl`` command.
    """
    description = "Developer tools for formatting, testing, and the documentation."
    dev = commands.add_parser("dev", help=description, description=description)
    tools = dev.add_subparsers(dest="tool", metavar="TOOL", required=True)

    description = "Format, lint, and type-check the code (ruff format, ruff check --fix, ty check)."
    parser = tools.add_parser("format", help=description, description=description)
    parser.set_defaults(func=format_code.run)

    description = "Run the test suite with pytest."
    parser = tools.add_parser("test", help=description, description=description)
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        metavar="PYTEST_ARGS",
        help="Arguments passed to pytest (e.g. tests/test_fuel.py -k density).",
    )
    parser.set_defaults(func=_test)

    description = "Build the HTML documentation with Sphinx."
    parser = tools.add_parser("build-docs", help=description, description=description)
    parser.set_defaults(func=build_docs.run)

    description = "Remove the Sphinx documentation build artifacts."
    parser = tools.add_parser("clean-docs", help=description, description=description)
    parser.set_defaults(func=clean_docs.run)

    description = "Open the built HTML documentation in a browser."
    parser = tools.add_parser("view-docs", help=description, description=description)
    parser.set_defaults(func=_view_docs)
