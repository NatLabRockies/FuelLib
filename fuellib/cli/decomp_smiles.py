"""Command-line SMILES decomposition for FuelLib group-contribution methods."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from fuellib.gcm import gani, nannoolal, unifac

DECOMPOSERS = {
    "gani": gani.decompose,
    "nannoolal": nannoolal.decompose,
    "unifac": unifac.decompose,
}


def run(args: argparse.Namespace) -> None:
    """Decompose supplied SMILES strings and optionally write group counts to CSV.

    Args:
        args: Parsed command-line arguments of ``fl decompose``.

    Raises:
        ValueError: If the SMILES sources are inconsistent or the CSV lacks a
            'SMILES' column.
    """
    if not args.smiles and not args.input:
        raise ValueError(
            "Provide one or more SMILES strings or an input CSV file to decompose."
        )
    if args.name and (not args.smiles or len(args.name) != len(args.smiles)):
        raise ValueError("Provide exactly one --name for each --smiles string.")

    decompose = DECOMPOSERS[args.method]

    labels: list[str] = []
    smiles_list: list[str] = []
    if args.smiles:
        labels.extend(args.name or args.smiles)
        smiles_list.extend(args.smiles)
    if args.input:
        input_df = pd.read_csv(args.input)
        if "SMILES" not in input_df.columns:
            raise ValueError("Input CSV must contain a 'SMILES' column.")
        names = input_df.get("Reference Compound", input_df["SMILES"])
        labels.extend(names.fillna(input_df["SMILES"]))
        smiles_list.extend(input_df["SMILES"])

    decomposed = []
    for label, smiles in zip(labels, smiles_list, strict=True):
        try:
            decomposed.append(decompose(smiles))
        except (KeyError, ValueError) as error:
            # decompose() reports the method, SMILES, and cause; add only the label.
            prefix = f"Error decomposing {label!r}: " if label != smiles else "Error: "

            print(f"{prefix}{error.args[0]}", file=sys.stderr)
            sys.exit(1)
    df = pd.concat(decomposed)
    df.index = pd.Index(labels, name="Reference Compound")

    if args.out:
        df.to_csv(args.out)
        print(f"Decomposition counts written to {args.out}")
    else:
        print(df.loc[:, (df != 0).any()].to_string())


def add_parser(commands: "argparse._SubParsersAction") -> None:
    """Add the ``decompose`` command.

    Args:
        commands: Subparsers action of the ``fl`` command.
    """
    description = "Decompose SMILES strings into group-contribution method groups."
    parser = commands.add_parser("decompose", help=description, description=description)
    parser.add_argument(
        "-s",
        "--smiles",
        nargs="+",
        help="One or more SMILES strings to decompose.",
    )
    parser.add_argument(
        "-n",
        "--name",
        nargs="+",
        help=(
            "Optional reference compound names, one for each SMILES string given "
            "with --smiles."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        metavar="CSV",
        help=(
            "Optional CSV file with a 'SMILES' column and an optional "
            "'Reference Compound' column."
        ),
    )
    parser.add_argument(
        "-m",
        "--method",
        default="gani",
        type=str.casefold,
        choices=DECOMPOSERS,
        help="Group-contribution method to use (default: gani).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        metavar="CSV",
        help="Optional CSV file for the decomposition counts (default: print).",
    )
    parser.set_defaults(func=run)
