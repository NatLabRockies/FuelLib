"""Command-line SMILES decomposition for FuelLib group-contribution methods."""

import argparse
import sys
from pathlib import Path
from typing import cast

import pandas as pd
from rdkit import Chem

from fuellib.gcm import gani, nannoolal, unifac

DECOMPOSERS = {
    "gani": gani.decompose,
    "nannoolal": nannoolal.decompose,
    "unifac": unifac.decompose,
}

_COLUMNS = {"smiles": "SMILES", "inchi": "InChI"}
"""Input CSV column holding the structures, by ``--from`` format."""


def _to_smiles(structure: object, source_format: str) -> str:
    """Convert a molecular structure to SMILES.

    Args:
        structure: Structure string (or a missing CSV value).
        source_format: Format of ``structure``, ``"smiles"`` or ``"inchi"``.

    Returns:
        The SMILES string. SMILES input is returned unchanged (even if it is a
        missing CSV value) so the decomposer reports the error with its context.

    Raises:
        ValueError: If an InChI is empty or cannot be parsed.
    """
    if source_format == "smiles":
        return cast(str, structure)
    if not isinstance(structure, str) or not structure.strip():
        raise ValueError("InChI must be a non-empty string.")
    mol = Chem.MolFromInchi(structure)
    if mol is None:
        raise ValueError(f"Could not parse InChI {structure!r}.")
    return Chem.MolToSmiles(mol)


def run(args: argparse.Namespace) -> None:
    """Decompose supplied structures and optionally write group counts to CSV.

    Args:
        args: Parsed command-line arguments of ``fl decomp``.

    Raises:
        ValueError: If the structure sources are inconsistent or the CSV lacks
            the structure column.
    """
    column = _COLUMNS[args.source_format]
    if not args.structures and not args.input:
        raise ValueError(
            "Provide one or more structures or an input CSV file to decompose."
        )
    if args.name and (not args.structures or len(args.name) != len(args.structures)):
        raise ValueError("Provide exactly one --name for each structure.")

    decompose = DECOMPOSERS[args.method]

    labels: list[str] = []
    structures: list[str] = []
    if args.structures:
        labels.extend(args.name or args.structures)
        structures.extend(args.structures)
    if args.input:
        input_df = pd.read_csv(args.input)
        if column not in input_df.columns:
            raise ValueError(f"Input CSV must contain a '{column}' column.")
        names = input_df.get("Reference Compound", input_df[column])
        labels.extend(names.fillna(input_df[column]))
        structures.extend(input_df[column])

    decomposed = []
    for label, structure in zip(labels, structures, strict=True):
        try:
            decomposed.append(decompose(_to_smiles(structure, args.source_format)))
        except (KeyError, ValueError) as error:
            # decompose() reports the method, SMILES, and cause; add only the label.
            prefix = (
                f"Error decomposing {label!r}: " if label != structure else "Error: "
            )
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
    """Add the ``decomp`` command.

    Args:
        commands: Subparsers action of the ``fl`` command.
    """
    description = (
        "Decompose molecular structures into group-contribution method groups."
    )
    parser = commands.add_parser("decomp", help=description, description=description)
    parser.add_argument(
        "-s",
        "--structures",
        "--smiles",
        nargs="+",
        metavar="STRUCTURE",
        help="One or more structures to decompose, in the format given by --from.",
    )
    parser.add_argument(
        "-n",
        "--name",
        nargs="+",
        help=(
            "Optional reference compound names, one for each structure given "
            "with --structures."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        metavar="CSV",
        help=(
            "Optional CSV file with a 'SMILES' (or 'InChI', see --from) column "
            "and an optional 'Reference Compound' column."
        ),
    )
    parser.add_argument(
        "--from",
        dest="source_format",
        default="smiles",
        type=str.casefold,
        choices=_COLUMNS,
        help="Format of the input structures (default: smiles).",
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
