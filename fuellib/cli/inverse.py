"""Command-line interface for inverse fuel design using FuelLib."""

try:
    from ..simulate.inverse import d7566_config, Problem, solve
except ImportError as e:
    raise SystemExit(f"Error: {e}") from None
from pathlib import Path
import pandas as pd
from ..fuel import Fuel
from ..data import DEFAULT_DATA_DIR

import argparse

pd.set_option("display.max_rows", None)

CONFIGS = {"d7566": d7566_config}


def main():
    parser = argparse.ArgumentParser(description="Inverse fuel design using FuelLib.")

    parser.add_argument(
        "-c",
        "--config",
        default="d7566",
        type=str.casefold,
        choices=CONFIGS.keys(),
        help="Configuration to use for the inverse fuel design.",
    )
    parser.add_argument(
        "-f",
        "--fuel-name",
        required=True,
        type=str,
        metavar="NAME",
        help="Name of the initial fuel.",
    )
    parser.add_argument(
        "-dir",
        "--fuel-data-dir",
        default=DEFAULT_DATA_DIR,
        type=Path,
        metavar="PATH",
        help="Directory where fuel data files are located (default: fuellib/data).",
    )
    parser.add_argument(
        "-m",
        "--max-steps",
        default=100_000,
        type=int,
        metavar="N",
        help="Maximum number of optimization steps (default: 100,000).",
    )
    parser.add_argument(
        "--disable-ref-props",
        dest="disable_ref_props",
        action="store_true",
        help="Do not use reference compound properties.",
    )
    parser.add_argument(
        "--ignore-composition",
        dest="ignore_composition",
        action="store_true",
        help=(
            "Ignore the original fuel composition ('Weight %%') and design a "
            "mixture of its components from scratch (additive fraction of 1)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        metavar="DIR",
        help=(
            "Directory to export the optimized GCxGC data, additive GCxGC data, "
            "and blend ratio to (default: None)."
        ),
    )
    args = parser.parse_args()

    fuel = Fuel(
        args.fuel_name,
        fuelDataDir=args.fuel_data_dir,
        useRefProperties=not args.disable_ref_props,
    )
    config = CONFIGS[args.config]
    problem = Problem(
        config=config, fuel=fuel, ignore_composition=args.ignore_composition
    )
    solution = solve(problem, max_steps=args.max_steps)

    if args.output is not None:
        solution.save(args.output, args.fuel_name)


if __name__ == "__main__":
    main()
