"""Command-line interface for FuelLib: the ``fl`` command and its subcommands."""

import argparse
from collections.abc import Sequence

from . import convert, decomp_smiles, dev, export, fuel_manager, plotting, simulate

_COMMANDS = (decomp_smiles, export, fuel_manager, plotting, simulate, convert, dev)
"""Modules that register a command with ``add_parser(commands)``."""


def _parser() -> argparse.ArgumentParser:
    """Build the parser of the ``fl`` command with all of its subcommands."""
    parser = argparse.ArgumentParser(
        prog="fl", description="FuelLib command-line interface."
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for module in _COMMANDS:
        module.add_parser(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``fl`` command.

    Args:
        argv: Command-line arguments without the program name. Defaults to
            ``sys.argv[1:]``.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"fl {args.command}: error: {error}\n")


if __name__ == "__main__":
    main()
