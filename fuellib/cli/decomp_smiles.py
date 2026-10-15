"""Command-line SMILES decomposition for FuelLib group-contribution methods."""

import argparse
from pathlib import Path

from ..gcm.decomposition import (
    UnsupportedGroupError,
    decompose,
    format_decomposition,
    read_decomposition_input,
    write_decompositions_csv,
)


def main() -> int:
    """Decompose supplied SMILES strings and optionally write group counts to CSV."""
    parser = argparse.ArgumentParser(
        description="Decompose SMILES strings into functional-group counts.",
        epilog="""Examples:
  %(prog)s "CCc1ccccc1"
  %(prog)s "CCc1ccccc1" -n ethylbenzene -o gani.csv
  %(prog)s -i fuel.csv -m gani -o gani.csv

Input CSV files require a SMILES column. An optional Reference Compound column is
written as Common_Name in the output CSV.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "smiles",
        nargs="*",
        metavar="SMILES",
        help="One or more SMILES strings to decompose.",
    )
    parser.add_argument(
        "-n",
        "--name",
        metavar="NAME",
        help="Optional name for one command-line SMILES.",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        metavar="CSV",
        help="CSV file containing SMILES.",
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
        "--output",
        "--output-file",
        type=Path,
        metavar="CSV",
        help="Optional CSV file for the decomposition counts.",
    )
    arguments = parser.parse_args()

    if arguments.name is not None and len(arguments.smiles) != 1:
        parser.error("--name requires exactly one command-line SMILES string.")
    if arguments.name is not None and arguments.input_file is not None:
        parser.error("--name cannot be used with --input-file.")

    compounds = [(arguments.name or "", smiles) for smiles in arguments.smiles]
    if arguments.input_file is not None:
        try:
            compounds.extend(read_decomposition_input(arguments.input_file))
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if not compounds:
        parser.error("Provide one or more SMILES strings or an input file.")

    decompositions: list[tuple[str, str, dict[str, int]]] = []
    for reference_compound, smiles in compounds:
        try:
            counts = decompose(smiles, method=arguments.method)
        except (UnsupportedGroupError, ValueError) as error:
            parser.error(f"Cannot decompose {smiles!r}: {error}")
        decompositions.append((reference_compound, smiles, counts))
        label = f"{reference_compound} ({smiles})" if reference_compound else smiles
        print(f"{label}: {format_decomposition(counts, method=arguments.method)}")

    if arguments.output is not None:
        write_decompositions_csv(
            arguments.output, decompositions, method=arguments.method
        )
        print(f"Wrote {len(decompositions)} decomposition(s) to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
