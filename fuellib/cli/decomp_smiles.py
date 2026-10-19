"""Command-line SMILES decomposition for FuelLib group-contribution methods."""

import argparse
from pathlib import Path

import pandas as pd

from fuellib import gcm


def decompose_smiles() -> None:
    """Decompose supplied SMILES strings and optionally write group counts to CSV."""
    parser = argparse.ArgumentParser()
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
        metavar="METHOD",
        help="Group-contribution method to use (default: gani).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        metavar="CSV",
        help="Optional CSV file for the decomposition counts (default: print).",
    )
    args = parser.parse_args()
    if not args.smiles and not args.input:
        parser.error(
            "Provide one or more SMILES strings or an input CSV file to decompose."
        )
    if args.name and (not args.smiles or len(args.name) != len(args.smiles)):
        parser.error("Provide exactly one --name for each --smiles string.")

    if args.method.casefold() == "gani":
        decompose = gcm.gani.decompose
    else:
        parser.error(f"Unsupported method: {args.method}")

    labels: list[str] = []
    smiles_list: list[str] = []
    if args.smiles:
        labels.extend(args.name or args.smiles)
        smiles_list.extend(args.smiles)
    if args.input:
        input_df = pd.read_csv(args.input)
        if "SMILES" not in input_df.columns:
            parser.error("Input CSV must contain a 'SMILES' column.")
        names = input_df.get("Reference Compound", input_df["SMILES"])
        labels.extend(names.fillna(input_df["SMILES"]))
        smiles_list.extend(input_df["SMILES"])

    df = pd.concat([decompose(smiles) for smiles in smiles_list])
    df.index = pd.Index(labels, name="Reference Compound")

    if args.out:
        df.to_csv(args.out)
        print(f"Decomposition counts written to {args.out}")
    else:
        print(df.loc[:, (df != 0).any()].to_string())


if __name__ == "__main__":
    decompose_smiles()
