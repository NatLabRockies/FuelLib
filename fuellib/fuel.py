"""Fuel class for Group Contribution Method calculations."""

import os
import re
from typing import cast

import numpy as np
import pandas as pd
import pint
from scipy.optimize import curve_fit

from ._data_locator import (
    get_fueldata_decomp_dir,
    get_fueldata_dir,
    get_fueldata_gc_dir,
    get_fueldata_props_dir,
    get_gcmtable_dir,
    get_metadata_decomp_name,
)
from .convert import K2C
from .units import Q_, ureg
from .utility import mixing_rule

#: Default ambient-gas parameters for diffusion_coeff (air).
_AIR_SIGMA = Q_(3.62e-10, "m")
_AIR_EPSILON_BY_KB = Q_(97.0, "K")
_AIR_MW = Q_(28.97e-3, "kg/mol")

# Expected physical dimensionality for each GCM table row, used to flag
# unexpected unit changes in gcmTable.csv (see get_row below).
_GCM_ROW_EXPECTED_UNITS = {
    "tck": "dimensionless",
    "pck": "bar ** -0.5",
    "vck": "m ** 3 / kmol",
    "tbk": "dimensionless",
    "tmk": "dimensionless",
    "hfk": "kJ/mol",
    "gfk": "kJ/mol",
    "hvk": "kJ/mol",
    "wk": "dimensionless",
    "vmk": "m ** 3 / kmol",
    "CpAk": "J/mol/K",
    "CpBk": "J/mol/K",
    "CpCk": "J/mol/K",
    "MW": "g/mol",
}


def _normalize_gcm_unit_text(raw_unit):
    """Normalize a gcmTable.csv 'Units' cell into a pint-parseable string."""
    text = str(raw_unit).strip()
    if text.casefold() == "dimensionless":
        return ""
    text = text.replace("^", " ** ")
    # CSV uses "KJ" for kilojoule; pint would otherwise read "K" as kelvin.
    text = re.sub(r"\bKJ\b", "kJ", text)
    return text


class fuel:
    """
    Class for handling group contribution calculations of thermodynamic and mixture properties.

    :param name: Name of the mixture as it appears in its gcData file.
    :type name: str
    :param decompName: Name of the groupDecomposition file if different from name. Defaults to None.
    :type decompName: str, optional
    :param fuelDataDir: Directory where the fuel data is stored. If None, uses built-in embedded data.
    :type fuelDataDir: str, optional
    """

    # Type annotations for documented attributes
    #: Root directory for fuel data (custom or embedded)
    fuelDataDir: str

    #: Directory containing GCxGC compositional data files
    fuelDataGcDir: str

    #: Directory containing functional group decomposition files
    fuelDataDecompDir: str

    #: Directory containing experimental property data (may be None)
    fuelDataPropsDir: str | None

    #: Name of the fuel/mixture
    name: str

    #: List of compound names in the mixture
    compounds: list

    #: Molecular formulas for each compound
    formulas: np.ndarray | None

    #: Mass fractions of each compound. Shape: (num_compounds,)
    Y_0: np.ndarray

    #: Functional group decomposition matrix. Shape: (num_compounds, num_groups)
    Nij: np.ndarray

    #: Number of compounds in the mixture
    num_compounds: int

    #: Number of functional groups in the decomposition
    num_groups: int

    #: Molecular weights. Shape: (num_compounds,)
    MW: pint.Quantity

    #: Critical temperatures. Shape: (num_compounds,)
    Tc: pint.Quantity

    #: Critical pressures. Shape: (num_compounds,)
    Pc: pint.Quantity

    #: Critical volumes. Shape: (num_compounds,)
    Vc: pint.Quantity

    #: Boiling temperatures. Shape: (num_compounds,)
    Tb: pint.Quantity

    #: Melting temperatures. Shape: (num_compounds,)
    Tm: pint.Quantity

    #: Enthalpy of formation. Shape: (num_compounds,)
    Hf: pint.Quantity

    #: Gibbs free energy. Shape: (num_compounds,)
    Gf: pint.Quantity

    #: Enthalpy of vaporization at 298 K. Shape: (num_compounds,)
    Hv_stp: pint.Quantity

    #: Latent heat of vaporization at 298 K. Shape: (num_compounds,)
    Lv_stp: pint.Quantity

    #: Molar specific heat at 298 K. Shape: (num_compounds,)
    Cp_stp: pint.Quantity

    #: Molar liquid volume at 298 K. Shape: (num_compounds,)
    Vm_stp: pint.Quantity

    #: Acentric factors (dimensionless). Shape: (num_compounds,)
    omega: pint.Quantity

    #: Lennard-Jones collision diameters. Shape: (num_compounds,)
    sigma: pint.Quantity

    #: Lennard-Jones well depths. Shape: (num_compounds,)
    epsilonByKB: pint.Quantity

    #: Hydrocarbon types ("n-alkane", "iso-alkane", "cyclo-alkane", "aromatic", "alkene")
    hc_type: np.ndarray

    #: Family codes for thermal conductivity (0: saturated, 1: aromatic, 2: cycloparaffin, 3: olefin)
    fam: np.ndarray

    #: Carbon numbers. Shape: (num_compounds,)
    nC: np.ndarray

    #: Hydrogen numbers. Shape: (num_compounds,)
    nH: np.ndarray

    #: PelePhysics keys for each compound (if available)
    pelephysics_keys: np.ndarray | None

    # Number of first and second order groups from Constantinou and Gani
    N_g1 = 78
    N_g2 = 43

    def __init__(self, name, decompName=None, fuelDataDir=None):
        """
        Initialize the fuel object and calculate GCM properties.

        :param name: Name of the mixture as it appears in its gcData file.
        :type name: str
        :param decompName: Name of the groupDecomposition file if different from name.
        :type decompName: str, optional
        :param fuelDataDir: Directory where the fuel data is stored. If None, uses built-in embedded data.
        :type fuelDataDir: str, optional
        """

        self.name = name
        if decompName is None:
            # Try to get decomposition name from metadata
            decompName = get_metadata_decomp_name(name, fuelDataDir)

        # Determine and set data directories for this fuel instance
        if fuelDataDir is None:
            # Use built-in embedded data
            self.fuelDataDir = get_fueldata_dir()
            self.fuelDataGcDir = get_fueldata_gc_dir()
            self.fuelDataDecompDir = get_fueldata_decomp_dir()
            self.fuelDataPropsDir = get_fueldata_props_dir()
        else:
            # Validate and use custom fuel directory
            from ._data_locator import (
                _get_props_dir_for_fueldata,
                _validate_fuel_data_dir,
            )

            _validate_fuel_data_dir(fuelDataDir)
            self.fuelDataDir = fuelDataDir
            self.fuelDataGcDir = os.path.join(fuelDataDir, "gcData")
            self.fuelDataDecompDir = os.path.join(fuelDataDir, "groupDecompositionData")
            self.fuelDataPropsDir = _get_props_dir_for_fueldata(fuelDataDir)

        # Get GCM table directory (always from built-in data)
        gcmtable_dir = get_gcmtable_dir()

        self.groupDecompFile = os.path.join(self.fuelDataDecompDir, f"{decompName}.csv")
        self.gcxgcFile = os.path.join(self.fuelDataGcDir, f"{name}_init.csv")
        self.gcmTableFile = os.path.join(gcmtable_dir, "gcmTable.csv")

        # Read functional group data for mixture (num_compounds,num_groups)
        df_Nij = pd.read_csv(self.groupDecompFile)
        self.Nij = df_Nij.iloc[:, 1:].to_numpy()
        self.num_compounds = self.Nij.shape[0]
        self.num_groups = self.Nij.shape[1]

        # Classify hydrocarbon by family (used in thermal conductivity)
        # 0: saturated hydrocarbons
        # 1: aromatics
        # 2: cycloparaffins
        # 3: olefins
        self.fam = np.zeros(self.num_compounds, dtype=int)

        # Classify hydrocarbon by type (n-alkane, iso-alkane, cyclo-alkane, aromatic)
        # Based on group decompositions from Constantinou-Gani method
        self.hc_type = np.array([""] * self.num_compounds, dtype=object)

        aromatics = 10  # starting index for aromatic groups
        num_aromatics = 5
        branching = 78  # starting index for branching groups (Group j (CH3)2CH through C(CH3)2C(CH3)2)
        num_branching = 5  # groups 78-82 inclusive
        cyclos = 83  # starting index for membered ring groups (3-7 membered rings)
        num_cyclos = 5
        olefins = 4  # starting index for double bound groups
        num_olefins = 6

        for i in range(self.num_compounds):
            # Check if aromatic: does it contain AC's?
            if sum(self.Nij[i, aromatics : aromatics + num_aromatics]) > 0:
                self.fam[i] = 1
                self.hc_type[i] = "aromatic"
            # Check if cycloparaffin: does it contain rings?
            elif sum(self.Nij[i, cyclos : cyclos + num_cyclos]) > 0:
                self.fam[i] = 2
                self.hc_type[i] = "cyclo-alkane"
            # Check if olefin: does it contain double bonds?
            elif sum(self.Nij[i, olefins : olefins + num_olefins]) > 0:
                self.fam[i] = 3
                self.hc_type[i] = "alkene"
            # Check for branching groups (CH, C quaternary carbons)
            elif sum(self.Nij[i, branching : branching + num_branching]) > 0:
                self.hc_type[i] = "iso-alkane"
            else:
                # Only CH3 and CH2 -> n-alkane (linear)
                self.hc_type[i] = "n-alkane"

        # Calculate carbon and hydrogen numbers from first-order group decomposition
        # For jet fuels, use only alkyl (0-3) and aromatic (10-14) groups
        # Alkyl: CH3=1C,3H; CH2=1C,2H; CH=1C,1H; C=1C,0H
        # Aromatic: ACH=1C,1H; AC=1C,0H; ACCH3=2C,3H; ACCH2=2C,2H; ACCH=2C,1H
        alkyl_carbons = np.array([1, 1, 1, 1])  # groups 0-3
        alkyl_hydrogens = np.array([3, 2, 1, 0])
        # Olefinic: group 4 appears to represent 2 carbons with 3 hydrogens in UNIFAC-based system
        olefinic_carbons = np.array([2, 1, 1, 0, 0, 0])  # groups 4-9
        olefinic_hydrogens = np.array([3, 1, 0, 0, 0, 0])
        aromatic_carbons = np.array([1, 1, 2, 2, 2])  # groups 10-14
        aromatic_hydrogens = np.array([1, 0, 3, 2, 1])

        self.nC = np.zeros(self.num_compounds, dtype=float)
        self.nH = np.zeros(self.num_compounds, dtype=float)
        for i in range(self.num_compounds):
            # Alkyl contribution (groups 0-3)
            self.nC[i] = np.dot(self.Nij[i, 0:4], alkyl_carbons)
            self.nH[i] = np.dot(self.Nij[i, 0:4], alkyl_hydrogens)
            # Olefinic contribution (groups 4-9)
            self.nC[i] += np.dot(self.Nij[i, 4:10], olefinic_carbons)
            self.nH[i] += np.dot(self.Nij[i, 4:10], olefinic_hydrogens)
            # Aromatic contribution (groups 10-14)
            self.nC[i] += np.dot(self.Nij[i, 10:15], aromatic_carbons)
            self.nH[i] += np.dot(self.Nij[i, 10:15], aromatic_hydrogens)

        # Read GCxGC/compound data
        df_gcxgc = pd.read_csv(self.gcxgcFile)

        self.compounds = [
            compound.strip() for compound in df_gcxgc["Compound"].to_list()
        ]

        # Load molecular formulas if available
        if "Formula" in df_gcxgc.columns:
            self.formulas = np.array(
                [
                    formula.strip() if pd.notna(formula) else None
                    for formula in df_gcxgc["Formula"].to_list()
                ]
            )
        else:
            self.formulas = None

        if "PelePhysics Key" in df_gcxgc.columns:
            self.pelephysics_keys = np.array(
                [key.strip() for key in df_gcxgc["PelePhysics Key"].to_list()]
            )
        else:
            self.pelephysics_keys = None

        self.Y_0 = df_gcxgc["Weight %"].to_numpy().flatten().astype(float)
        self.Y_0 /= np.sum(self.Y_0)

        # Make sure mixture data is consistent:
        if self.num_groups < self.N_g1:
            raise ValueError(
                f"Insufficient mixture description:\n"
                f"The number of columns in {self.groupDecompFile} is less than "
                f"the required number of first-order groups (N_g1 = {self.N_g1})."
            )
        if self.Y_0.shape[0] != self.num_compounds:
            raise ValueError(
                f"Insufficient mixture description:\n"
                f"The number of compounds in {self.groupDecompFile} does not "
                f"equal the number of compounds in {self.gcxgcFile}."
            )

        # Read and store GCM table properties
        df_table = pd.read_csv(self.gcmTableFile)

        def get_row(property_name):
            """
            Get property row from GCM table as a unit-aware Quantity.

            Units are read from the GCM table's "Units" column and validated
            against the dimensionality this calculation expects, so drift in
            the source CSV is flagged instead of silently mis-computed.

            :param property_name: Name of the property to retrieve.
            :type property_name: str
            :return: Property values for all functional groups.
            :rtype: pint.Quantity
            :raises ValueError: If property not found, or its units in the
                GCM table don't match the expected dimensionality.
            """
            row = df_table[df_table["Property"] == property_name]
            if row.empty:
                raise ValueError(f"Property '{property_name}' not found in GCM table.")

            csv_unit_text = row["Units"].iloc[0]
            unit_str = _normalize_gcm_unit_text(csv_unit_text)
            try:
                parsed_unit = ureg.parse_units(unit_str)
            except pint.UndefinedUnitError as e:
                raise ValueError(
                    f"GCM table property '{property_name}' has unparseable "
                    f"units '{csv_unit_text}' in {self.gcmTableFile}."
                ) from e

            expected_unit_str = _GCM_ROW_EXPECTED_UNITS[property_name]
            expected_dim = ureg.parse_units(expected_unit_str).dimensionality
            if parsed_unit.dimensionality != expected_dim:
                raise ValueError(
                    f"GCM table property '{property_name}' has units "
                    f"'{csv_unit_text}' ({parsed_unit.dimensionality}) but this "
                    f"calculation expects dimensionality {expected_dim} (e.g. "
                    f"'{expected_unit_str}'). Check {self.gcmTableFile} for "
                    f"unexpected changes."
                )

            values = row.iloc[:, 2:].to_numpy().flatten().astype(float)
            return Q_(values, parsed_unit)

        # Table data for functional groups (num_compounds,)
        Tck = get_row("tck")  # critical temperature correlation input
        Pck = get_row("pck")  # critical pressure correlation input
        Vck = get_row("vck")  # critical volume
        Tbk = get_row("tbk")  # boiling temperature correlation input
        Tmk = get_row("tmk")  # melting point temperature correlation input
        hfk = get_row("hfk")  # enthalpy of formation
        gfk = get_row("gfk")  # Gibbs energy
        hvk = get_row("hvk")  # latent heat of vaporization
        wk = get_row("wk")  # accentric factor correlation input
        Vmk = get_row("vmk")  # liquid molar volume fraction
        cpak = get_row("CpAk")  # specific heat values
        cpbk = get_row("CpBk")  # specific heat values
        cpck = get_row("CpCk")  # specific heat values
        mwk = get_row("MW")  # molecular weights

        # --- Compute critical properties at standard temp (num_compounds,)
        # Molecular weights
        self.MW = (self.Nij @ mwk).to("kg/mol")

        # T_c (critical temperature)
        self.Tc = Q_(181.128, "K") * np.log((self.Nij @ Tck).magnitude)

        # p_c (critical pressure)
        self.Pc = (
            Q_(1.3705, "bar") + (self.Nij @ Pck + Q_(0.10022, "bar ** -0.5")) ** (-2)
        ).to("Pa")

        # V_c (critical volume)
        self.Vc = (Q_(-0.00435, "m ** 3 / kmol") + self.Nij @ Vck).to("m ** 3 / mol")

        # T_b (boiling temperature)
        self.Tb = Q_(204.359, "K") * np.log((self.Nij @ Tbk).magnitude)

        # T_m (melting temperature)
        self.Tm = Q_(102.425, "K") * np.log((self.Nij @ Tmk).magnitude)

        # H_f (enthalpy of formation)
        self.Hf = (Q_(10.835, "kJ/mol") + self.Nij @ hfk).to("J/mol")

        # G_f (Gibbs free energy)
        self.Gf = (Q_(-14.828, "kJ/mol") + self.Nij @ gfk).to("J/mol")

        # H_v,stp (enthalpy of vaporization at 298 K)
        self.Hv_stp = (Q_(6.829, "kJ/mol") + self.Nij @ hvk).to("J/mol")

        # omega (accentric factor)
        self.omega = cast(
            pint.Quantity,
            Q_(
                0.4085 * np.log((self.Nij @ wk).magnitude + 1.1507) ** (1.0 / 0.5050),
                "dimensionless",
            ),
        )

        # V_m (molar liquid volume at 298 K)
        self.Vm_stp = (Q_(0.01211, "m ** 3 / kmol") + self.Nij @ Vmk).to("m ** 3 / mol")

        # C_p,stp (molar specific heat at 298 K)
        self.Cp_stp = (self.Nij @ cpak) - Q_(19.7779, "J/mol/K")

        # Temperature corrections for C_p
        self.Cp_B = self.Nij @ cpbk
        self.Cp_C = self.Nij @ cpck

        # L_v,stp (latent heat of vaporization at 298 K)
        self.Lv_stp = (self.Hv_stp / self.MW).to("J/kg")

        # Lennard-Jones parameters for diffusion calculations (Tee et al. 1966)
        self.epsilonByKB = ((0.7915 + 0.1693 * self.omega) * self.Tc).to("K")

        # Tee et al. (1966) correlation requires Tc[K]/Pc[atm] magnitudes and
        # yields sigma in Angstroms (not a dimensionally homogeneous formula).
        Tc_K = self.Tc.to("K").magnitude
        Pc_atm = self.Pc.to("atm").magnitude
        sigma_angstrom = (2.3551 - 0.0874 * self.omega.magnitude) * (Tc_K / Pc_atm) ** (
            1.0 / 3
        )
        self.sigma = cast(pint.Quantity, Q_(sigma_angstrom, "angstrom").to("m"))

    # -------------------------------------------------------------------------
    # Member functions
    # -------------------------------------------------------------------------
    def mean_molecular_weight(self, Yi):
        """
        Calculate the mean molecular weight of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :return: Mean molecular weight of the mixture.
        :rtype: pint.Quantity
        """
        if np.sum(Yi) != 0:
            Mbar = 1 / np.sum(Yi / self.MW)  # mean molar weight of the mixture
        else:
            Mbar = Q_(0.0, self.MW.units)

        return Mbar

    def mass2Y(self, mass):
        """
        Calculate the mass fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: pint.Quantity
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Normalize to get group mole fractions
        total_mass = np.sum(mass)
        if total_mass != 0:
            Yi = (mass / total_mass).to("dimensionless").magnitude
        else:
            Yi = np.zeros_like(self.MW.magnitude)

        return Yi

    def mass2X(self, mass):
        """
        Calculate the mole fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: pint.Quantity
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Calculate the number of moles for each compound
        num_mole = mass / self.MW

        # Normalize to get group mole fractions
        total_moles = np.sum(num_mole)
        if total_moles != 0:
            Xi = (num_mole / total_moles).to("dimensionless").magnitude
        else:
            Xi = np.zeros_like(self.MW.magnitude)

        return Xi

    def X2Y(self, Xi):
        """
        Calculate the mass fractions from the mole fractions of each component.

        :param Xi: Mole fractions of each compound.
        :type Xi: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Calculate the mass for each compound
        mass = Xi * self.MW

        # Normalize to get group mass fractions
        total_mass = np.sum(mass)
        if total_mass != 0:
            Yi = (mass / total_mass).to("dimensionless").magnitude
        else:
            Yi = np.zeros_like(self.MW.magnitude)

        return Yi

    def Y2X(self, Yi):
        """
        Calculate the mole fractions from the mass fractions of each component.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :return: Mole fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        Mbar = self.mean_molecular_weight(Yi)
        if np.sum(Yi) != 0:
            Xi = (Mbar * Yi / self.MW).to("dimensionless").magnitude
        else:
            Xi = np.zeros_like(self.MW.magnitude)

        return Xi

    def density(self, T, comp_idx=None):
        """
        Calculate the density of each component at temperature T.

        :param T: Temperature of the mixture.
        :type T: pint.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Density of each compound.
        :rtype: pint.Quantity
        """
        if comp_idx is None:
            MW = self.MW
            Vm = self.molar_liquid_vol(T)
        else:
            MW = self.MW[comp_idx]
            Vm = self.molar_liquid_vol(T, comp_idx=comp_idx)

        rho = (MW / Vm).to("kg / m ** 3")
        return rho

    def viscosity_kinematic(self, T, comp_idx=None):
        """
        Calculate the viscosity using Dutt's equation.

        :meta private: This uses Dutt's equation (4.23) from "Viscosity of Liquids".
        :meta private: The equation predicts viscosity in mm^2/s and is converted to SI units.

        :param T: Temperature.
        :type T: pint.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Viscosity of each component.
        :rtype: pint.Quantity
        """

        # Dutt's equation (4.23) is an empirical fit requiring magnitudes in
        # degrees Celsius (not dimensionally homogeneous).
        T_cels = K2C(T).magnitude
        if comp_idx is None:
            Tb_cels = K2C(self.Tb).magnitude
        else:
            Tb_cels = K2C(self.Tb[comp_idx]).magnitude

        # RHS of Dutt's equation (4.23) in Viscosity of Liquids
        rhs = -3.0171 + (442.78 + 1.6452 * Tb_cels) / (T_cels + 239 - 0.19 * Tb_cels)
        nu_i = np.exp(rhs)  # Viscosity in mm^2/s

        return Q_(nu_i, "mm ** 2 / s").to("m ** 2 / s")

    def viscosity_dynamic(self, T, comp_idx=None):
        """
        Calculate liquid dynamic viscosity based on droplet temperature and density.

        :meta private: Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

        :param T: Temperature.
        :type T: pint.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Dynamic viscosity.
        :rtype: pint.Quantity
        """

        nu_i = self.viscosity_kinematic(T, comp_idx=comp_idx)
        rho_i = self.density(T, comp_idx=comp_idx)
        mu_i = (nu_i * rho_i).to("Pa * s")
        return mu_i

    def Cp(self, T, comp_idx=None):
        """
        Compute molar specific heat capacity at a given temperature.

        :param T: Temperature.
        :type T: pint.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar specific heat capacity.
        :rtype: pint.Quantity
        """

        theta = ((T.to("K") - Q_(298.0, "K")) / Q_(700.0, "K")).to("dimensionless")
        if comp_idx is None:
            Cp_stp = self.Cp_stp
            Cp_B = self.Cp_B
            Cp_C = self.Cp_C
        else:
            Cp_stp = self.Cp_stp[comp_idx]
            Cp_B = self.Cp_B[comp_idx]
            Cp_C = self.Cp_C[comp_idx]

        cp = Cp_stp + Cp_B * theta + Cp_C * theta**2

        return cp

    def Cl(self, T, comp_idx=None):
        """
        Compute liquid mass specific heat capacity at a given temperature.

        :param T: Temperature.
        :type T: pint.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Mass specific heat capacity.
        :rtype: pint.Quantity
        """
        if comp_idx is None:
            MW = self.MW
        else:
            MW = self.MW[comp_idx]
        cp = self.Cp(T, comp_idx=comp_idx)
        return cp / MW

    def psat(self, T, comp_idx=None, correlation="Lee-Kesler"):
        """
        Compute saturated vapor pressure.

        :meta private: Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

        :param T: Temperature.
        :type T: pint.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Saturated vapor pressure.
        :rtype: pint.Quantity
        """

        if comp_idx is None:
            Tr = (T / self.Tc).to("dimensionless")
            Pc = self.Pc
            omega = self.omega
        else:
            Tr = (T / self.Tc[comp_idx]).to("dimensionless")
            Pc = self.Pc[comp_idx]
            omega = self.omega[comp_idx]

        if correlation.casefold() == "Ambrose-Walton".casefold():
            # May cause trouble at high temperatures
            tau = 1 - Tr
            f0 = (
                -5.97616 * tau
                + 1.29874 * tau**1.5
                - 0.60394 * tau**2.5
                - 1.06841 * tau**5.0
            )
            f0 /= Tr
            f1 = (
                -5.03365 * tau
                + 1.11505 * tau**1.5
                - 5.41217 * tau**2.5
                - 7.46628 * tau**5.0
            )
            f1 /= Tr
            f2 = (
                -0.64771 * tau
                + 2.41539 * tau**1.5
                - 4.26979 * tau**2.5
                - 3.25259 * tau**5.0
            )
            f2 /= Tr
            rhs = np.exp(f0 + omega * f1 + omega**2 * f2)

        else:  # Default correlation is Lee-Kesler
            f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
            f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
            rhs = np.exp(f0 + omega * f1)

        psat = Pc * rhs
        return psat

    def psat_antoine_coeffs(self, Tvals=None, units="mks", correlation="Lee-Kesler"):
        """
        Estimate Antoine coefficients for vapor pressure of an individual compound.

        :param Tvals: Temperature range or nodes for Antoine fit (default [273.15 K, Tb_i]).
        :type Tvals: pint.Quantity, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: 4 np.ndarrays
        """

        # Define or get temperature nodes for fit (Kelvin magnitudes, for
        # scipy.optimize.curve_fit which requires plain floats)
        if Tvals is None:
            print("Tvals not specified, using [273.15, Tb_i] for each compound.")
            # Initialize as zeros for now, calculated for each compound later
            T_K = np.zeros(20)
        elif len(Tvals) == 2:
            T_K = np.linspace(
                Tvals[0].to("K").magnitude, Tvals[1].to("K").magnitude, 20
            )
        elif len(Tvals) > 2:
            T_K = Tvals.to("K").magnitude
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(T, A, B, C):
            """Antoine equation for vapor pressure."""
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = 1  # default is Pa
        if units.lower() == "bar":
            D = 1e5
        elif units.lower() == "atm":
            D = 1.01325e5
        elif units.lower() == "cgs":
            D = 1 / 10  # dyne/cm^2

        # Fit Antoine coefficients for each compound
        A = np.zeros(self.num_compounds)
        B = np.zeros(self.num_compounds)
        C = np.zeros(self.num_compounds)
        for i in range(self.num_compounds):
            # Update T if not specified
            if Tvals is None:
                T_K = np.linspace(273.15, self.Tb[i].to("K").magnitude, 20)
            Pvals = np.zeros_like(T_K)
            for k in range(len(T_K)):
                Pvals[k] = (
                    1
                    / D
                    * self.psat(Q_(T_K[k], "K"), correlation=correlation)[i]
                    .to("Pa")
                    .magnitude
                )

            logP = np.log10(Pvals)
            popt, _ = curve_fit(antoine_eq, T_K, logP, p0=[1, 1e3, -1])
            A[i], B[i], C[i] = popt
        D = D + np.zeros(self.num_compounds)  # make D an array
        return A, B, C, D

    def molar_liquid_vol(self, T, comp_idx=None):
        """
        Compute molar liquid volume with temperature correction.

        :param T: Temperature.
        :type T: pint.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar liquid volume.
        :rtype: pint.Quantity
        """

        Tstp = Q_(298.0, "K")
        if comp_idx is None:
            Tc = self.Tc
            omega = self.omega
            Vm_stp = self.Vm_stp
        else:
            Tc = self.Tc[[comp_idx]]
            omega = self.omega[[comp_idx]]
            Vm_stp = self.Vm_stp[[comp_idx]]
        phi = np.zeros(len(Tc))
        for i in range(len(Tc)):
            if T > Tc[i]:
                phi[i] = -(
                    ((1 - (Tstp / Tc[i])).to("dimensionless").magnitude) ** (2.0 / 7.0)
                )
            else:
                phi[i] = (
                    ((1 - (T / Tc[i])).to("dimensionless").magnitude) ** (2.0 / 7.0)
                ) - (
                    ((1 - (Tstp / Tc[i])).to("dimensionless").magnitude) ** (2.0 / 7.0)
                )
        z = 0.29056 - 0.08775 * omega.magnitude
        Vmi = Vm_stp * np.power(z, phi)
        if comp_idx is not None:
            Vmi = Vmi[0]
        return Vmi

    def latent_heat_vaporization(self, T, comp_idx=None):
        """
        Calculate latent heat of vaporization adjusted for temperature.

        :param T: Temperature.
        :type T: pint.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Latent heat of vaporization.
        :rtype: pint.Quantity
        """
        if comp_idx is None:
            Tc = self.Tc
            Tb = self.Tb
            Lv_stp = self.Lv_stp
        else:
            Tc = self.Tc[[comp_idx]]
            Tb = self.Tb[[comp_idx]]
            Lv_stp = self.Lv_stp[[comp_idx]]

        # Reduced temperatures
        Tr = (T / Tc).to("dimensionless").magnitude
        Trb = (Tb / Tc).to("dimensionless").magnitude

        Lv_unit = Lv_stp.units
        Lv_mag = Lv_stp.magnitude
        Lvi_mag = np.zeros(len(Tc))
        for i in range(len(Tc)):
            if T > Tc[i]:
                Lvi_mag[i] = 0.0
            else:
                Lvi_mag[i] = Lv_mag[i] * (((1.0 - Tr[i]) / (1.0 - Trb[i])) ** 0.38)
        Lvi = cast(pint.Quantity, Q_(Lvi_mag, Lv_unit))

        if comp_idx is not None:
            Lvi = Lvi[0]
        return Lvi

    def diffusion_coeff(
        self,
        p,
        T,
        sigma_gas=_AIR_SIGMA,
        epsilonByKB_gas=_AIR_EPSILON_BY_KB,
        MW_gas=_AIR_MW,
        correlation="Tee",
    ):
        """
        Compute diffusion coefficients using Lennard-Jones parameters.

        :meta private: Uses Wilke and Lee method (Poling, equation 11-4.1).
        :meta private: Ambient gas defaults to air parameters.

        :param p: Pressure.
        :type p: pint.Quantity
        :param T: Temperature.
        :type T: pint.Quantity
        :param sigma_gas: Collision diameter.
        :type sigma_gas: pint.Quantity, optional
        :param epsilonByKB_gas: Well depth over Boltzmann constant.
        :type epsilonByKB_gas: pint.Quantity, optional
        :param MW_gas: Mean molecular weight of ambient gas.
        :type MW_gas: pint.Quantity, optional
        :param correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
        :type correlation: str, optional
        :return: Diffusion coefficient.
        :rtype: pint.Quantity
        """

        # Poling (11-3.x/11-4.x) is a unit-specific empirical correlation:
        # magnitudes must be in Angstroms, K, g/mol, and bar; the result comes
        # out in cm^2/s.
        if correlation.casefold() == "Tee".casefold():
            sigma_i = self.sigma.to("angstrom").magnitude
            epsilonByKB_i = self.epsilonByKB.to("K").magnitude
        else:
            # Method of Wilke & Lee calculating liquid sigma and epsilon
            Vmb_i = np.zeros(self.num_compounds)
            for n in range(self.num_compounds):
                Vmb_i[n] = (
                    self.molar_liquid_vol(self.Tb[n])[n].to("cm ** 3 / mol").magnitude
                )
            sigma_i = 1.18 * Vmb_i ** (1 / 3)  # Angstroms, Poling (11-4.2)
            epsilonByKB_i = 1.15 * self.Tb.to("K").magnitude  # K , Poling (11-4.3)

        # Compute binary sigma and epsilon
        sigma_gas_ang = sigma_gas.to("angstrom").magnitude
        epsilonByKB_gas_K = epsilonByKB_gas.to("K").magnitude
        sigmaAB_i = (sigma_gas_ang + sigma_i) / 2  # Angstroms, Poling (11-3.5)
        epsilonAB_byKB_i = (
            epsilonByKB_gas_K * epsilonByKB_i
        ) ** 0.5  # K, Poling (11-3.4)

        # Dimensionless collision integral for diffusion: Poling (11-3.6)
        T_K = T.to("K").magnitude
        Tstar_i = T_K / epsilonAB_byKB_i  # [1]
        A = 1.06036
        B = 0.15610
        C = 0.193
        D = 0.47635
        E = 1.03587
        F = 1.52996
        G = 1.76474
        H = 3.89411
        omegaD_i = (
            A / (Tstar_i**B)
            + C / np.exp(D * Tstar_i)
            + E / np.exp(F * Tstar_i)
            + G / np.exp(H * Tstar_i)
        )

        # Molecular weights and pressure in the units the correlation expects
        MW_gas_g = MW_gas.to("g/mol").magnitude
        MW_i_g = self.MW.to("g/mol").magnitude
        M_AB_i = 2 * (MW_i_g * MW_gas_g) / (MW_i_g + MW_gas_g)  # g/mol, Poling (11-3.1)

        p_bar = p.to("bar").magnitude

        # Binary diffusion coefficients, Poling (11-4.1)
        D_AB_i_cm2s = (
            1e-3
            * (3.03 - 0.98 / (M_AB_i**0.5))
            * (T_K**1.5)
            / (p_bar * M_AB_i**0.5 * sigmaAB_i**2 * omegaD_i)
        )  # cm^2/s

        return Q_(D_AB_i_cm2s, "cm ** 2 / s").to("m ** 2 / s")

    def surface_tension(self, T, comp_idx=None, correlation="Brock-Bird"):
        """
        Calculate surface tension of each compound at a given temperature.

        :meta private: Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

        :param T: Temperature.
        :type T: pint.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Brock-Bird" or "Pitzer").
        :type correlation: str, optional
        :return: Surface tension.
        :rtype: pint.Quantity
        """
        if comp_idx is None:
            Tc = self.Tc
            Pc = self.Pc
            Tb = self.Tb
            omega = self.omega
        else:
            Tc = self.Tc[[comp_idx]]
            Pc = self.Pc[[comp_idx]]
            Tb = self.Tb[[comp_idx]]
            omega = self.omega[[comp_idx]]

        # Brock-Bird/Pitzer (Poling 12-3.5/12-3.7) is a unit-specific empirical
        # correlation: magnitudes must be in bar and K; result comes out in
        # dyn/cm.
        Tr = (T / Tc).to("dimensionless").magnitude
        Pc_bar = Pc.to("bar").magnitude
        Tc_K = Tc.to("K").magnitude

        if correlation.casefold() == "Brock-Bird".casefold():
            Tbr = (Tb / Tc).to("dimensionless").magnitude
            Qc = 0.1196 * (1.0 + (Tbr * np.log(Pc_bar / 1.01325)) / (1.0 - Tbr)) - 0.279
        else:
            w = omega.magnitude
            Qc = (
                (1.86 + 1.18 * w)
                / 19.05
                * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
            )

        st_dyncm = (
            Pc_bar ** (2.0 / 3.0) * Tc_K ** (1.0 / 3.0) * Qc * (1 - Tr) ** (11.0 / 9.0)
        )
        st = cast(pint.Quantity, Q_(st_dyncm, "dyn/cm").to("N/m"))

        if comp_idx is not None:
            st = st[0]

        return st

    def thermal_conductivity(self, T, comp_idx=None):
        """
        Calculate thermal conductivity at a given temperature.

        :meta private: Uses Latini et al. method (Poling equation 10-9.1).

        :param T: Temperature.
        :type T: pint.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Thermal conductivity.
        :rtype: pint.Quantity
        """
        if comp_idx is None:
            MW = self.MW
            Tc = self.Tc
            Tb = self.Tb
            fam = self.fam
        else:
            MW = self.MW[[comp_idx]]
            Tc = self.Tc[[comp_idx]]
            Tb = self.Tb[[comp_idx]]
            fam = np.array([self.fam[comp_idx]])

        # Latini (Poling 10-9.1) is a unit-specific empirical correlation:
        # magnitudes must be in g/mol and K; result comes out in W/m/K.
        MW_g = MW.to("g/mol").magnitude
        Tb_K = Tb.to("K").magnitude
        Tc_K = Tc.to("K").magnitude
        Tr = (T / Tc).to("dimensionless").magnitude

        Astar = 0.00350 + np.zeros_like(Tc_K)
        alpha = 1.2
        beta = 0.5 + np.zeros_like(Tc_K)
        gamma = 0.167
        MW_beta = MW_g.copy()

        for i in range(len(Tc_K)):
            if fam[i] == 1:
                # Aromatics
                Astar[i] = 0.0346
                beta[i] = 1.0
            elif fam[i] == 2:
                # Cycloparaffins
                Astar[i] = 0.0310
                beta[i] = 1.0
            elif fam[i] == 3:
                # Olefins
                Astar[i] = 0.0361
                beta[i] = 1.0
            MW_beta[i] = MW_beta[i] ** beta[i]

        A = Astar * Tb_K**alpha / (MW_beta * Tc_K**gamma)
        tc_val = A * (1 - Tr) ** (0.38) / (Tr ** (1 / 6))
        tc = cast(pint.Quantity, Q_(tc_val, "W / m / K"))

        if comp_idx is not None:
            tc = tc[0]
        return tc

    # --- Mixture functions ---
    def mixture_density(self, Yi, T):
        """
        Calculate mixture density at a given temperature.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature.
        :type T: pint.Quantity
        :return: Mixture density.
        :rtype: pint.Quantity
        """
        MW = self.MW  # Molecular weights of each component
        Vmi = self.molar_liquid_vol(T)  # Molar volume of each component

        # Calculate density
        rho = (Yi @ (MW / Vmi)).to("kg / m ** 3")

        return rho

    def mixture_kinematic_viscosity(self, Yi, T, correlation="Kendall-Monroe"):
        """
        Calculate kinematic viscosity of the mixture.

        :meta private: Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature.
        :type T: pint.Quantity
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture kinematic viscosity.
        :rtype: pint.Quantity
        """
        nu_i = self.viscosity_kinematic(T)  # Viscosities of individual components

        # Calculate mole fractions for each species
        Xi = self.Y2X(Yi)

        if correlation.casefold() == "Arrhenius".casefold():
            # Arrhenius log-mixing rule requires a numeric magnitude in a
            # fixed unit (m^2/s), since log of a dimensioned value is undefined.
            nu_mag = np.exp(np.sum(Xi * np.log(nu_i.to("m ** 2 / s").magnitude)))
            nu = Q_(nu_mag, "m ** 2 / s")
        else:
            # Default: Kendall-Monroe mixing correlation
            nu = np.sum(Xi * (nu_i ** (1.0 / 3.0))) ** (3.0)

        return nu

    def mixture_dynamic_viscosity(self, Yi, T, correlation="Kendall-Monroe"):
        """
        Calculate dynamic viscosity of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature.
        :type T: pint.Quantity
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture dynamic viscosity.
        :rtype: pint.Quantity
        """

        nu = self.mixture_kinematic_viscosity(Yi, T, correlation=correlation)
        rho = self.mixture_density(Yi, T)

        return (rho * nu).to("Pa * s")

    def mixture_vapor_pressure(self, Yi, T, correlation="Lee-Kesler"):
        """
        Calculate vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature.
        :type T: pint.Quantity
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Mixture vapor pressure.
        :rtype: pint.Quantity
        """

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Saturated vapor pressure for each compound
        p_sati = self.psat(T, correlation=correlation)

        # Mixture vapor pressure via Raoult's law
        p_v = p_sati @ Xi

        return p_v

    def mixture_vapor_pressure_antoine_coeffs(
        self, Yi, Tvals=None, units="mks", correlation="Lee-Kesler"
    ):
        """
        Estimate Antoine coefficients for vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param Tvals: Temperature range or nodes for Antoine fit (default [273.15 K, min(Tb)]).
        :type Tvals: pint.Quantity, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: float
        """

        # Define or get temperature nodes for fit (Kelvin magnitudes, for
        # scipy.optimize.curve_fit which requires plain floats)
        if Tvals is None:
            print("Tvals not specified, using [273.15, min(Tb_mix)] for mixture.")
            X = self.Y2X(Yi)
            Tb_mix_K = mixing_rule(self.Tb, X).to("K").magnitude
            T_K = np.linspace(273.15, Tb_mix_K, 20)
        elif len(Tvals) == 2:
            T_K = np.linspace(
                Tvals[0].to("K").magnitude, Tvals[1].to("K").magnitude, 20
            )
        elif len(Tvals) > 2:
            T_K = Tvals.to("K").magnitude
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(T, A, B, C):
            """
            Antoine equation for vapor pressure.

            :param T: Temperature.
            :type T: float or np.ndarray
            :param A: Antoine coefficient A.
            :type A: float
            :param B: Antoine coefficient B.
            :type B: float
            :param C: Antoine coefficient C.
            :type C: float
            :return: log10(pressure).
            :rtype: float or np.ndarray
            """
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = 1  # default is Pa
        if units.lower() == "bar":
            D = 1e5
        elif units.lower() == "atm":
            D = 1.01325e5
        elif units.lower() == "cgs":
            D = 1 / 10  # dyne/cm^2

        Pvals = np.zeros_like(T_K)
        for k in range(len(T_K)):
            Pvals[k] = (
                self.mixture_vapor_pressure(
                    Yi, Q_(T_K[k], "K"), correlation=correlation
                )
                .to("Pa")
                .magnitude
                / D
            )

        logP = np.log10(Pvals)
        popt, _ = curve_fit(antoine_eq, T_K, logP, p0=[1, 1e3, -1])  # initial guess
        A, B, C = popt

        return A, B, C, D

    def mixture_surface_tension(self, Yi, T, correlation="Brock-Bird"):
        """
        Calculate surface tension of the mixture.

        :meta private: Uses arithmetic pseudo-property method recommended by Hugill and van Welsenes (1986).

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature.
        :type T: pint.Quantity
        :param correlation: Correlation method ("Pitzer" or "Brock-Bird").
        :type correlation: str, optional
        :return: Mixture surface tension.
        :rtype: pint.Quantity
        """

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Surface tension for each compound
        sti = self.surface_tension(T, correlation=correlation)

        # Mixture surface tension via arithmetic mean, Poling (12-5.2)
        st = mixing_rule(sti, Xi, "arithmetic")

        return st

    def mixture_thermal_conductivity(self, Yi, T):
        """
        Calculate thermal conductivity of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature.
        :type T: pint.Quantity
        :return: Thermal conductivity.
        :rtype: pint.Quantity
        """
        tc = self.thermal_conductivity(T)
        return np.sum(Yi * tc ** (-2)) ** (-0.5)


__all__ = ["fuel"]
