"""Export fuel properties for use in CFD simulations (CONVERGE and PelePhysics)."""

import argparse
import os

import fuellib as fl

from ..exporters.converge import FUELDATA_DIR, export_converge
from ..exporters.pele import export_pele


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by all exporters."""
    parser.add_argument(
        "-f",
        "--fuel-name",
        required=True,
        metavar="NAME",
        help="Name of the fuel (mandatory).",
    )
    parser.add_argument(
        "-dir",
        "--fuel-data-dir",
        default=FUELDATA_DIR,
        metavar="PATH",
        help="Directory where fuel data files are located (optional, default: fuellib/data).",
    )
    parser.add_argument(
        "-o",
        "--export-dir",
        default=os.getcwd(),
        metavar="PATH",
        help="Directory to export the properties (optional, default: current working directory).",
    )


def _converge(args: argparse.Namespace) -> None:
    """Export fuel properties for CONVERGE simulations."""
    print("Preparing to export mixture properties:")
    print(f"    Fuel name: {args.fuel_name}")
    if args.export_mix:
        print("    Exporting mixture properties: True")
    print("    Units: mks")
    print(f"    Minimum temperature: {args.temp_min} K")
    print(f"    Maximum temperature: {args.temp_max} K")
    print(f"    Temperature step size: {args.temp_step} K")
    print(f"    Export directory: {args.export_dir}")
    print(f"    Fuel data directory: {args.fuel_data_dir}")

    fuel = fl.Fuel(args.fuel_name, data_dir=args.fuel_data_dir)
    export_converge(
        fuel,
        path=args.export_dir,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_step=args.temp_step,
        export_mix=args.export_mix,
    )

    print("\nExport completed successfully!")


def _pele(args: argparse.Namespace) -> None:
    """Export fuel properties for PelePhysics simulations."""
    units = args.units.lower()
    liq_prop_model = args.liq_prop_model.lower()
    dep_fuel_names = args.dep_fuel_names

    print("Preparing to export properties:")
    print(f"    Fuel name: {args.fuel_name}")
    print(f"    Units: {units}")
    print(f"    Liquid property model: {liq_prop_model}")
    if liq_prop_model == "mp":
        print(f"    Antoine coefficients: {args.psat_antoine}")
    print(f"    Export mixture properties: {args.export_mix}")
    print(f"    Export directory: {args.export_dir}")
    print(f"    Fuel data directory: {args.fuel_data_dir}")

    fuel = fl.Fuel(args.fuel_name, data_dir=args.fuel_data_dir)
    if dep_fuel_names is None and not args.export_mix:
        if args.use_pp_keys and fuel.comps.pelephysics_keys is not None:
            dep_fuel_names = list(fuel.comps.pelephysics_keys)
        else:
            dep_fuel_names = (
                fuel.gc_data["GC-Bin"].tolist()
                if "GC-Bin" in fuel.gc_data.columns
                else list(fuel.comps.names)
            )

    export_pele(
        fuel,
        path=args.export_dir,
        units=units,
        dep_fuel_names=dep_fuel_names,
        use_pp_keys=args.use_pp_keys,
        export_mix=args.export_mix,
        export_mix_name=args.export_mix_name,
        liq_prop_model=liq_prop_model,
        psat_antoine=args.psat_antoine,
    )

    print("\nExport completed successfully!")


def add_parser(commands: "argparse._SubParsersAction") -> None:
    """Add the ``export`` command and its subcommands.

    Args:
        commands: Subparsers action of the ``fl`` command.
    """
    description = "Export fuel properties for CFD simulations."
    export = commands.add_parser("export", help=description, description=description)
    targets = export.add_subparsers(dest="target", metavar="TARGET", required=True)

    description = "Export mixture fuel properties for CONVERGE simulations."
    parser = targets.add_parser("converge", help=description, description=description)
    _add_common_arguments(parser)
    parser.add_argument(
        "-t",
        "--temp-min",
        type=float,
        default=0,
        metavar="K",
        help="Minimum temperature for property calculations (optional, default: 0).",
    )
    parser.add_argument(
        "-T",
        "--temp-max",
        type=float,
        default=1000,
        metavar="K",
        help="Maximum temperature for property calculations (optional, default: 1000).",
    )
    parser.add_argument(
        "-s",
        "--temp-step",
        type=int,
        default=10,
        metavar="K",
        help="Step size for temperature (optional, default: 10).",
    )
    parser.add_argument(
        "-m",
        "--export-mix",
        action="store_true",
        help="Export mixture properties instead of component properties.",
    )
    parser.set_defaults(func=_converge)

    description = "Export fuel properties for PelePhysics simulations."
    parser = targets.add_parser("pele", help=description, description=description)
    _add_common_arguments(parser)
    parser.add_argument(
        "-u",
        "--units",
        default="mks",
        metavar="{mks,cgs}",
        help="Units for critical properties (optional, default: mks).",
    )
    parser.add_argument(
        "-dep",
        "--dep-fuel-names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Space-separated list or single fuel that each compound deposits to (optional, default: fuel.comps.names).",
    )
    parser.add_argument(
        "-pp",
        "--use-pp-keys",
        action="store_true",
        help="Use PelePhysics keys for each compound.",
    )
    parser.add_argument(
        "-m",
        "--export-mix",
        action="store_true",
        help="Export mixture properties of the fuel.",
    )
    parser.add_argument(
        "-mn",
        "--export-mix-name",
        default=None,
        metavar="NAME",
        help="Name the mixture if different than fuel_name (optional, default: fuel_name).",
    )
    parser.add_argument(
        "-l",
        "--liq-prop-model",
        default="gcm",
        metavar="{gcm,mp}",
        help="Model for liquid properties (optional, default: gcm).",
    )
    parser.add_argument(
        "-psat",
        "--psat-antoine",
        action="store_true",
        help="Use Antoine coefficients for vapor pressure in the MP model.",
    )
    parser.set_defaults(func=_pele)
