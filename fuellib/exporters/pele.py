import os
import pandas as pd
import argparse
import subprocess
from datetime import datetime
from scipy import stats as st
import fuellib as fl

# Default data directory - use fuellib's embedded data
FUELDATA_DIR = fl.get_fueldata_dir()

"""
Script that exports critical properties and initial mass fraction data
for use in Pele simulations.

This script is designed to be run from the command line and will create
a file named "sprayPropsGCM_<fuel_name>.inp" or "sprayPropsMP_<fuel_name>.inp"
in the specified directory. The file contains properties for each compound in 
the fuel, formatted for Pele.

Usage:
    fl-export-pele -f <fuel_name>

For detailed options, run:
    fl-export-pele -h
"""


class UnitConverter:
    """Unit conversion factors for different unit systems used in Pele exports."""

    def __init__(self, units: str):
        """
        Initialize converter for specified unit system.

        :param units: Unit system ('cgs' or 'mks').
        :type units: str
        """
        self.units = units.lower()
        self._validate_units()
        self._set_conversion_factors()

    def _validate_units(self):
        """
        Validate that the unit system is supported.

        :raises ValueError: If unit system is not 'mks' or 'cgs'.
        """
        if self.units not in ["mks", "cgs"]:
            raise ValueError(f"Units must be 'mks' or 'cgs', got '{self.units}'")

    def _set_conversion_factors(self):
        """
        Set conversion factors based on unit system.
        """
        if self.units == "cgs":
            # Convert from MKS to CGS
            self.MW = 1e3  # kg/mol to g/mol
            self.Cp = 1e4  # J/kg/K to erg/g/K
            self.Vm = 1e6  # m^3/mol to cm^3/mol
            self.Lv = 1e4  # J/kg to erg/g
            self.P = 1e1  # Pa to dyne/cm^2
        else:
            # MKS units (no conversion)
            self.MW = 1.0
            self.Cp = 1.0
            self.Vm = 1.0
            self.Lv = 1.0
            self.P = 1.0


def get_git_info():
    """
    Get git commit hash and remote URL for file header.

    :return: Tuple containing git commit hash and remote URL.
    :rtype: tuple[str, str]
    """
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except Exception:
        git_commit = "N/A"

    try:
        git_remote = (
            subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            .strip()
            .decode("utf-8")
        )
    except Exception:
        git_remote = "N/A"

    return git_commit, git_remote


def get_filename(fuel_name, liq_prop_model, export_mix, path):
    """
    Generate appropriate filename based on parameters.

    :param fuel_name: Name of the fuel.
    :type fuel_name: str
    :param liq_prop_model: Liquid property model ('gcm' or 'mp').
    :type liq_prop_model: str
    :param export_mix: Whether exporting mixture properties.
    :type export_mix: bool
    :param path: Directory path for output file.
    :type path: str
    :return: Full path to output file.
    :rtype: str
    """
    if liq_prop_model.lower() == "gcm":
        if not export_mix:
            return os.path.join(path, f"sprayPropsGCM_{fuel_name}.inp")
        else:
            return os.path.join(path, f"sprayPropsGCM_mixture_{fuel_name}.inp")
    else:  # mp method
        if not export_mix:
            return os.path.join(path, f"sprayPropsMP_{fuel_name}.inp")
        else:
            return os.path.join(path, f"sprayPropsMP_mixture_{fuel_name}.inp")


def create_individual_compounds_dataframe(fuel, compound_names, converter):
    """
    Create DataFrame for individual compound properties.

    :param fuel: Fuel object containing compound properties.
    :type fuel: FuelLib.Fuel
    :param compound_names: List of compound names.
    :type compound_names: list[str]
    :param converter: Unit converter instance.
    :type converter: UnitConverter
    :return: DataFrame with compound properties.
    :rtype: pd.DataFrame
    """
    # Terms for liquid specific heat capacity in (J/kg/K) or (erg/g/K)
    # Cp(T) = Cp_A + Cp_B * theta + Cp_C * theta^2
    # where theta = (T - 298.15) / 700
    Cp_A = fuel.Cp_stp / fuel.MW
    Cp_B = fuel.Cp_B / fuel.MW
    Cp_C = fuel.Cp_C / fuel.MW

    return pd.DataFrame(
        {
            "Compound": compound_names,
            "Family": fuel.fam,
            "Y_0": fuel.Y_0,
            "MW": fuel.MW * converter.MW,
            "Tc": fuel.Tc,
            "Pc": fuel.Pc * converter.P,
            "Vc": fuel.Vc * converter.Vm,
            "Tb": fuel.Tb,
            "omega": fuel.omega,
            "Vm_stp": fuel.Vm_stp * converter.Vm,
            "Cp_A": Cp_A * converter.Cp,
            "Cp_B": Cp_B * converter.Cp,
            "Cp_C": Cp_C * converter.Cp,
            "Cp_stp": Cp_A * converter.Cp,  # For PeleMP model
            "Lv_stp": fuel.Lv_stp * converter.Lv,
        }
    )


def create_mixture_dataframe(fuel, export_mix_name, converter):
    """
    Create DataFrame for mixture properties.

    :param fuel: Fuel object containing mixture properties.
    :type fuel: FuelLib.Fuel
    :param export_mix_name: Name for the exported mixture.
    :type export_mix_name: str or None
    :param converter: Unit converter instance.
    :type converter: UnitConverter
    :return: DataFrame with mixture properties.
    :rtype: pd.DataFrame
    """
    if export_mix_name is None:
        export_mix_name = fuel.name
    if "posf" in export_mix_name.lower():
        export_mix_name = export_mix_name.upper()

    # Terms for liquid specific heat capacity in (J/kg/K) or (erg/g/K)
    # Cp(T) = Cp_A + Cp_B * theta + Cp_C * theta^2
    # where theta = (T - 298.15) / 700
    X = fuel.Y2X(fuel.Y_0)
    Cp_A = fl.utility.mixing_rule(fuel.Cp_stp / fuel.MW, X)
    Cp_B = fl.utility.mixing_rule(fuel.Cp_B / fuel.MW, X)
    Cp_C = fl.utility.mixing_rule(fuel.Cp_C / fuel.MW, X)

    return pd.DataFrame(
        {
            "Compound": [export_mix_name],
            "Family": [st.mode(fuel.fam).mode],
            "Y_0": [1.0],
            "MW": [fuel.mean_molecular_weight(fuel.Y_0) * converter.MW],
            "Tc": [fl.utility.mixing_rule(fuel.Tc, X)],
            "Pc": [fl.utility.mixing_rule(fuel.Pc, X) * converter.P],
            "Vc": [fl.utility.mixing_rule(fuel.Vc, X) * converter.Vm],
            "Tb": [fl.utility.mixing_rule(fuel.Tb, X)],
            "omega": [fl.utility.mixing_rule(fuel.omega, X)],
            "Vm_stp": [fl.utility.mixing_rule(fuel.Vm_stp, X) * converter.Vm],
            "Cp_A": [Cp_A * converter.Cp],
            "Cp_B": [Cp_B * converter.Cp],
            "Cp_C": [Cp_C * converter.Cp],
            "Cp_stp": [Cp_A * converter.Cp],  # For MP model: Cp_stp = Cp_A
            "Lv_stp": [fl.utility.mixing_rule(fuel.Lv_stp, X) * converter.Lv],
        }
    )


def vec_to_str(vec):
    """
    Convert a list or numpy array to a string representation.

    :param vec: List or numpy array to convert.
    :type vec: list or pd.Series or pd.DataFrame
    :return: String representation of the vector.
    :rtype: str
    """

    # If strings return string[0] string[1] ... string[n]
    if isinstance(vec, list):
        return " ".join(f"{v}" for v in vec)
    # Else if numbers, format with spaces between no commas or []
    elif isinstance(vec, (pd.Series, pd.DataFrame)):
        return " ".join(f"{v}" for v in vec.values)


def export_pele(
    fuel,
    path=None,
    units="mks",
    dep_fuel_names=None,
    use_pp_keys=True,
    export_mix=False,
    export_mix_name=None,
    liq_prop_model="gcm",
    psat_antoine=True,
):
    """
    Export fuel properties to input file for Pele simulations.

    :param fuel: An instance of the fuel class.
    :type fuel: fuel object

    :param path: Directory to save the input file.
    :type path: str, optional (default: current directory/exportData)

    :param units: Units for the properties ("mks" for SI, "cgs" for CGS).
    :type units: str, optional (default: "mks")

    :param dep_fuel_names: List or single fuel that each compound deposits to.
    :type dep_fuel_names: list of str, optional (default: None)

    :param use_pp_keys: Use the PelePhysics key for each compound (True or False). Default is False.
    :type use_pp_keys: bool, optional

    :param export_mix: Option to export mixture properties of the fuel (True or False).
    :type export_mix: bool, optional (default: False)

    :param export_mix_name: Name the mixture if different than fuel_name.
    :type export_mix_name: str, optional (default: None)

    :param liq_prop_model: Model for liquid properties. Options are "gcm" (default) or "mp".
    :type liq_prop_model: str, optional (default: "gcm")

    :param psat_antoine: Use Antoine coefficients for vapor pressure in MP model (True or False). Default is True.
    :type psat_antoine: bool, optional

    :return: None
    :rtype: None

    :raises ValueError: If input parameters are invalid
    :raises TypeError: If fuel object is not a FuelLib fuel instance
    """
    if path is None:
        path = os.path.join(os.getcwd(), "exportData")

    # Input validation
    if not hasattr(fuel, "compounds") or not hasattr(fuel, "Y_0"):
        raise TypeError("fuel parameter must be a valid FuelLib fuel object")

    if liq_prop_model.lower() not in ["gcm", "mp"]:
        raise ValueError(
            f"liq_prop_model must be 'gcm' or 'mp', got '{liq_prop_model}'"
        )

    # Initialize unit converter (also validates units)
    converter = UnitConverter(units)

    # Ensure output directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Generate output filename
    file_name = get_filename(fuel.name, liq_prop_model, export_mix, path)

    # Check if PelePhysics keys are available
    if use_pp_keys:
        if fuel.pelephysics_keys is not None:
            print(
                "\nPelePhysics keys found in GCxGC data, please ensure consistency with PelePhysics mechanism."
            )
            compound_names = fuel.pelephysics_keys
        else:
            print(
                "\nWarning: PelePhysics keys not found in GCxGC data. Using compound names instead."
            )
            compound_names = fuel.compounds
    else:
        if fuel.pelephysics_keys is not None:
            print(
                "\nWarning: PelePhysics keys found in GCxGC data, but not used. Using compound names instead."
            )
        compound_names = fuel.compounds

    # Check there are no spaces in compound_names
    for compound in compound_names:
        if " " in compound:
            raise ValueError(
                f"Pele cannot accept compounds with spaces. "
                f"Compound '{compound}' contains spaces. Use a '-' instead."
            )

    # Unit conversion factors are now handled by the UnitConverter class

    if not export_mix:
        print(
            f"\nCalculating GCM properties for individual compounds in {fuel.name}..."
        )

        # Validate and setup deposition fuel names
        if dep_fuel_names is None:
            dep_fuel_names = compound_names
        elif len(dep_fuel_names) == 1:
            # If a single deposition fuel name is provided, use it for all compounds
            dep_fuel_names = [dep_fuel_names[0]] * len(compound_names)
        elif len(dep_fuel_names) != len(compound_names):
            raise ValueError(
                "Length of dep_fuel_names must be one or match the number of compounds in the fuel."
            )

        # Create DataFrame with all properties and unit conversions
        df = create_individual_compounds_dataframe(fuel, compound_names, converter)

    else:
        print("\nCalculating mixture GCM properties at standard conditions...")

        # Create DataFrame with mixture properties and unit conversions
        df = create_mixture_dataframe(fuel, export_mix_name, converter)

        # Get the actual compound name from the DataFrame (may be modified by create_mixture_dataframe)
        compound_names = df["Compound"].tolist()

        # Setup mixture parameters
        if dep_fuel_names is None:
            dep_fuel_names = compound_names

    # Specific properties required for GCM method
    if liq_prop_model.lower() == "gcm":
        # Get the property names
        prop_names = [
            "Family",
            "MW",
            "Tc",
            "Pc",
            "Vc",
            "Tb",
            "omega",
            "Vm_stp",
            "Cp_A",
            "Cp_B",
            "Cp_C",
            "Lv_stp",
        ]

    else:  # mp method
        prop_names = ["MW", "Tc", "Tb", "Lv_stp", "Cp_stp", "rho"]
        if psat_antoine:
            prop_names.append("psat")

        # Calculate density at 298.15 K
        ref_T = 298.15
        if export_mix:
            rho = fuel.mixture_density(fuel.Y_0, ref_T)
        else:
            rho = fuel.density(ref_T)
        df["rho"] = rho

        # Get Antoine coefficients
        if psat_antoine:
            if export_mix:
                (
                    psat_A,
                    psat_B,
                    psat_C,
                    psat_D,
                ) = fuel.mixture_vapor_pressure_antoine_coeffs(fuel.Y_0, units=units)
                rho = fuel.mixture_density(fuel.Y_0, ref_T)
            else:
                psat_A, psat_B, psat_C, psat_D = fuel.psat_antoine_coeffs(units=units)
                rho = fuel.density(ref_T)

            df["psat_A"] = psat_A
            df["psat_B"] = psat_B
            df["psat_C"] = psat_C
            df["psat_D"] = psat_D

    # Dictionary of formatted names
    formatted_names = {
        "Family": ("family", ["", ""]),
        "MW": ("molar_weight", ["kg/mol", "g/mol"]),
        "Tc": ("crit_temp", ["K", "K"]),
        "Pc": ("crit_press", ["Pa", "dyne/cm^2"]),
        "Vc": ("crit_vol", ["m^3/mol", "cm^3/mol"]),
        "Tb": ("boil_temp", ["K", "K"]),
        "omega": ("acentric_factor", ["-", "-"]),
        "Vm_stp": ("molar_vol", ["m^3/mol", "cm^3/mol"]),
        "Cp_A": ("cp_a", ["J/kg/K", "erg/g/K"]),  # for GCM model
        "Cp_B": ("cp_b", ["J/kg/K", "erg/g/K"]),  # for GCM model
        "Cp_C": ("cp_c", ["J/kg/K", "erg/g/K"]),  # for GCM model
        "Cp_stp": ("cp", ["J/kg/K", "erg/g/K"]),  # for PeleMP model
        "Lv_stp": ("latent", ["J/kg", "erg/g"]),
        "rho": ("rho", ["kg/m^3", "g/cm^3"]),
        "psat": ("psat", ["Pa", "dyne/cm^2"]),
    }

    # Get header information
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    git_commit, git_remote = get_git_info()

    # Write the properties to the input file
    print(f"Writing properties to {file_name}.")
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, "a") as f:
        f.write(
            f"# -----------------------------------------------------------------------------\n"
            f"# Liquid fuel properties for {liq_prop_model.upper()} in Pele\n"
            f"# Fuel: {fuel.name}\n"
            f"# Number of compounds: {len(compound_names)}\n"
            f"# Generated: {dt_string}\n"
            f"# FuelLib remote URL: {git_remote}\n"
            f"# Git commit: {git_commit}\n"
            f"# Units: {units.upper()}\n"
            f"# -----------------------------------------------------------------------------\n\n"
        )
        f.write(f"particles.fuel_species = {vec_to_str(df['Compound'].tolist())}\n")
        f.write(f"particles.Y_0 = {vec_to_str(df['Y_0'].tolist())}\n")
        f.write(f"particles.dep_fuel_species = {vec_to_str(dep_fuel_names)}\n")
        if liq_prop_model.lower() == "mp":
            f.write(f"particles.fuel_ref_temp = {ref_T} # K\n")

        for comp_name in compound_names:
            f.write(f"\n# Properties for {comp_name} in {units.upper()}\n")
            for prop in prop_names:
                if prop in formatted_names:
                    prop_name, unit_txt = formatted_names[prop]
                    if units.lower() == "cgs":
                        unit_txt = unit_txt[1]
                    else:
                        unit_txt = unit_txt[0]
                    # MP model: Write Cp_stp as 'cp' for each component
                    if liq_prop_model.lower() == "mp" and prop == "Cp_stp":
                        value = df.loc[df["Compound"] == comp_name, prop].values[0]
                        f.write(
                            f"particles.{comp_name}_cp = {value:.6f} # {unit_txt}\n"
                        )
                        continue
                    # Write the property to the file
                    if prop == "Family":
                        value = df.loc[df["Compound"] == comp_name, prop].values[0]
                        if value == 0:
                            unit_txt = "saturated hydrocarbons"
                        elif value == 1:
                            unit_txt = "aromatics"
                        elif value == 2:
                            unit_txt = "cycloparaffins"
                        else:
                            unit_txt = "olefins"
                        f.write(
                            f"particles.{comp_name}_{prop_name} = {value} # {unit_txt}\n"
                        )
                    elif prop == "psat":
                        A = df.loc[df["Compound"] == comp_name, "psat_A"].values[0]
                        B = df.loc[df["Compound"] == comp_name, "psat_B"].values[0]
                        C = df.loc[df["Compound"] == comp_name, "psat_C"].values[0]
                        D = df.loc[df["Compound"] == comp_name, "psat_D"].values[0]
                        psat_coeffs = [A, B, C, D]
                        f.write(
                            f"particles.{comp_name}_{prop_name} = {vec_to_str(psat_coeffs)} # {unit_txt}\n"
                        )
                    elif not (liq_prop_model.lower() == "mp" and prop == "Cp_stp"):
                        value = df.loc[df["Compound"] == comp_name, prop].values[0]
                        f.write(
                            f"particles.{comp_name}_{prop_name} = {value:.6f} # {unit_txt}\n"
                        )


def main():
    """
    Main function to execute the export process.

    :param --fuel_name: Name of the fuel (mandatory).
    :type --fuel_name: str

    :param --fuel_data_dir: Directory where fuel data files are located. Default is FuelLib/fuelData.
    :type --fuel_data_dir: str, optional

    :param --fuel_decomp_name: Name of the decomposition file (optional). If not provided, defaults to fuel_name.
    :type --fuel_decomp_name: str, optional

    :param --units: Units for critical properties. Options are "mks" (default) or "cgs".
    :type --units: str, optional

    :param --dep_fuel_names: Space-separated list with len(fuel.compounds) or single fuel that all compounds deposit. Default is fuel.compounds.
    :type --dep_fuel_names: str, optional

    :param --use_pp_keys: Use the PelePhysics key for each compound (True or False). Default is True.
    :type --use_pp_keys: bool, optional

    :param --export_dir: Directory to export the properties. Default is "current directory/exportData".
    :type --export_dir: str, optional

    :param --export_mix: Option to export mixture properties of the fuel (True or False). Default is False.
    :type --export_mix: bool, optional

    :param --export_mix_name: Name the mixture if different than fuel_name. Default is fuel_name.
    :type --export_mix_name: str, optional

    :param --liq_prop_model: Model for liquid properties. Options are "gcm" (default) or "mp".
    :type --liq_prop_model: str, optional

    :param --psat_antoine: Use Antoine coefficients for vapor pressure in MP model (True or False). Default is True.
    :type --psat_antoine: bool, optional

    :raises FileNotFoundError: If required files for the specified fuel are not found.
    """

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Export fuel properties for Pele simulations."
    )

    # Mandatory argument for fuel name
    parser.add_argument(
        "-f",
        "--fuel_name",
        required=True,
        metavar="NAME",
        help="Name of the fuel (mandatory).",
    )

    # Optional argument for fuel data directory
    parser.add_argument(
        "-dir",
        "--fuel_data_dir",
        default=FUELDATA_DIR,
        metavar="PATH",
        help="Directory where fuel data files are located (optional, default: FuelLib/fuelData).",
    )

    # Optional argument for decomposition file name
    parser.add_argument(
        "-decomp",
        "--fuel_decomp_name",
        default=None,
        metavar="NAME",
        help="Name of the decomposition file (optional). If not provided, defaults to fuel_name.",
    )

    # Optional argument for units
    # Default is 'mks', but can be set to 'cgs'
    parser.add_argument(
        "-u",
        "--units",
        default="mks",
        metavar="{mks,cgs}",
        help="Units for critical properties (optional, default: mks).",
    )

    # Optional argument for deposition fuel names
    parser.add_argument(
        "-dep",
        "--dep_fuel_names",
        nargs="+",  # Accepts one or more values
        default=None,
        metavar="NAME",
        help="Space-separated list or single fuel that each compound deposits to (optional, default: fuel.compounds).",
    )

    # Optional argument for using PelePhysics key
    parser.add_argument(
        "-pp",
        "--use_pp_keys",
        type=lambda x: str(x).lower() in ["true", "1"],
        default=True,
        metavar="{true,false}",
        help="Use PelePhysics keys for each compound (optional, default: true).",
    )

    # Optional argument for export directory
    parser.add_argument(
        "-o",
        "--export_dir",
        default=os.path.join(os.getcwd(), "exportData"),
        metavar="PATH",
        help="Directory to export the properties (optional, default: ./exportData).",
    )

    # Optional argument for exporting mixture properties
    parser.add_argument(
        "-m",
        "--export_mix",
        type=lambda x: str(x).lower() in ["true", "1"],
        default=False,
        metavar="{true,false}",
        help="Export mixture properties of the fuel (optional, default: false).",
    )

    # Optional argument for mixture name if different than fuel_name
    parser.add_argument(
        "-mn",
        "--export_mix_name",
        default=None,
        metavar="NAME",
        help="Name the mixture if different than fuel_name (optional, default: fuel_name).",
    )

    # Optional argument for liquid property model
    parser.add_argument(
        "-l",
        "--liq_prop_model",
        default="gcm",
        metavar="{gcm,mp}",
        help="Model for liquid properties (optional, default: gcm).",
    )

    # Optional argument for printing Antoine coefficients in MP model
    parser.add_argument(
        "-psat",
        "--psat_antoine",
        type=lambda x: str(x).lower() in ["true", "1"],
        default=True,
        metavar="{true,false}",
        help="Use Antoine coefficients for vapor pressure in MP model (optional, default: true).",
    )

    # Parse arguments
    args = parser.parse_args()
    fuel_name = args.fuel_name
    fuel_decomp_name = args.fuel_decomp_name
    fuel_data_dir = args.fuel_data_dir

    # If decomposition name not provided, read from metadata (required)
    if fuel_decomp_name is None:
        fuel_decomp_name = fl.get_metadata_decomp_name(fuel_name, fuel_data_dir)

    units = args.units.lower()
    dep_fuel_names = args.dep_fuel_names
    use_pp_keys = args.use_pp_keys
    export_dir = args.export_dir
    export_mix = args.export_mix
    export_mix_name = args.export_mix_name
    liq_prop_model = args.liq_prop_model.lower()
    psat_antoine = args.psat_antoine

    # Print the parsed arguments
    print(f"Preparing to export properties:")
    print(f"    Fuel name: {fuel_name}")
    print(f"    Decomposition name: {fuel_decomp_name}")
    print(f"    Units: {units}")
    print(f"    Liquid property model: {liq_prop_model}")
    if liq_prop_model.lower() == "mp":
        print(f"    Antoine coefficients: {psat_antoine}")
    print(f"    Export mixture properties: {export_mix}")
    print(f"    Export directory: {export_dir}")
    print(f"    Fuel data directory: {fuel_data_dir}")

    # Create the groupContribution object for the specified fuel
    fuel = fl.fuel(fuel_name, decompName=fuel_decomp_name, fuelDataDir=fuel_data_dir)

    # Export properties for Pele
    export_pele(
        fuel,
        path=export_dir,
        units=units,
        dep_fuel_names=dep_fuel_names,
        use_pp_keys=use_pp_keys,
        export_mix=export_mix,
        export_mix_name=export_mix_name,
        liq_prop_model=liq_prop_model,
        psat_antoine=psat_antoine,
    )

    print("\nExport completed successfully!")


if __name__ == "__main__":
    main()
