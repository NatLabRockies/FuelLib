"""Fuel class for Group Contribution Method calculations."""

import os
from collections.abc import Sequence
from typing import Literal

import astropy.units.typing as ut
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
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
from .units import convert_temperature, ustrip
from .utility import mixing_rule

type FloatArrayLike = npt.NDArray[np.float64] | Sequence[float]


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
    fuelDataPropsDir: str

    #: Name of the fuel/mixture
    name: str

    #: List of compound names in the mixture
    compounds: list[str]

    #: Molecular formulas for each compound
    formulas: npt.NDArray[np.str_] | None

    #: Mass fractions of each compound. Shape: (num_compounds,)
    Y_0: npt.NDArray[np.float64]

    #: Functional group decomposition matrix. Shape: (num_compounds, num_groups)
    Nij: npt.NDArray[np.int_]

    #: Number of compounds in the mixture
    num_compounds: int

    #: Number of functional groups in the decomposition
    num_groups: int

    #: Molecular weights in kg/mol. Shape: (num_compounds,)
    MW: u.Quantity

    #: Critical temperatures in K. Shape: (num_compounds,)
    Tc: u.Quantity

    #: Critical pressures in Pa. Shape: (num_compounds,)
    Pc: u.Quantity

    #: Critical volumes in m³/mol. Shape: (num_compounds,)
    Vc: u.Quantity

    #: Boiling temperatures in K. Shape: (num_compounds,)
    Tb: u.Quantity

    #: Melting temperatures in K. Shape: (num_compounds,)
    Tm: u.Quantity

    #: Enthalpy of formation in J/mol. Shape: (num_compounds,)
    Hf: u.Quantity

    #: Gibbs free energy in J/mol. Shape: (num_compounds,)
    Gf: u.Quantity

    #: Enthalpy of vaporization at 298 K in J/mol. Shape: (num_compounds,)
    Hv_stp: u.Quantity

    #: Latent heat of vaporization at 298 K in J/kg. Shape: (num_compounds,)
    Lv_stp: u.Quantity

    #: Molar specific heat at 298 K in J/mol/K. Shape: (num_compounds,)
    Cp_stp: u.Quantity

    #: Molar liquid volume at 298 K in m³/mol. Shape: (num_comp ounds,)
    Vm_stp: u.Quantity

    #: Acentric factors. Shape: (num_compounds,)
    omega: u.Quantity

    #: Lennard-Jones collision diameters in m. Shape: (num_compounds,)
    sigma: u.Quantity

    #: Lennard-Jones well depths in K. Shape: (num_compounds,)
    epsilonByKB: u.Quantity

    #: Hydrocarbon types ("n-alkane", "iso-alkane", "cyclo-alkane", "aromatic", "alkene")
    hc_type: npt.NDArray[np.str_]

    #: Family codes for thermal conductivity (0: saturated, 1: aromatic, 2: cycloparaffin, 3: olefin)
    fam: npt.NDArray[np.int_]

    #: Carbon numbers. Shape: (num_compounds,)
    nC: npt.NDArray[np.int_]

    #: Hydrogen numbers. Shape: (num_compounds,)
    nH: npt.NDArray[np.int_]

    #: PelePhysics keys for each compound (if available)
    pelephysics_keys: npt.NDArray[np.str_] | None

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
        df_table = df_table.drop(columns=["Units"])

        def get_row(property_name):
            """
            Get property row from GCM table.

            :param property_name: Name of the property to retrieve.
            :type property_name: str
            :return: Property values for all functional groups.
            :rtype: np.ndarray
            :raises ValueError: If property not found in GCM table.
            """
            row = df_table[df_table["Property"] == property_name]
            if row.empty:
                raise ValueError(f"Property '{property_name}' not found in GCM table.")
            return row.iloc[:, 1:].to_numpy().flatten()

        # Table data for functional groups (num_compounds,)
        Tck = get_row("tck")  # critical temperature (1)
        Pck = get_row("pck")  # critical pressure (bar)
        Vck = get_row("vck")  # critical volume (m^3/kmol)
        Tbk = get_row("tbk")  # boiling temperature (1)
        Tmk = get_row("tmk")  # melting point temperature (1)
        hfk = get_row("hfk")  # enthalpy of formation, (kJ/mol)
        gfk = get_row("gfk")  # Gibbs energy (kJ/mol)
        hvk = get_row("hvk")  # latent heat of vaporization (kJ/mol)
        wk = get_row("wk")  # accentric factor (1)
        Vmk = get_row("vmk")  # liquid molar volume fraction (m^3/kmol)
        cpak = get_row("CpAk")  # specific heat values (J/mol/K)
        cpbk = get_row("CpBk")  # specific heat values (J/mol/K)
        cpck = get_row("CpCk")  # specific heat values (J/mol/K)
        mwk = get_row("MW")  # molecular weights (g/mol)

        # --- Compute critical properties at standard temp (num_compounds,)
        # Molecular weights
        _MW = np.matmul(self.Nij, mwk)  # g/mol
        self.MW = u.Quantity(_MW, "g/mol").to("kg/mol")

        # T_c (critical temperature)
        _Tc = 181.128 * np.log(np.matmul(self.Nij, Tck))  # K
        self.Tc = u.Quantity(_Tc, "K")

        # p_c (critical pressure)
        _Pc = 1.3705 + (np.matmul(self.Nij, Pck) + 0.10022) ** (-2)  # bar
        self.Pc = u.Quantity(_Pc, "bar").to("Pa")

        # V_c (critical volume)
        _Vc = -0.00435 + (np.matmul(self.Nij, Vck))  # m^3/kmol
        self.Vc = u.Quantity(_Vc, "m^3/kmol").to("m^3/mol")

        # T_b (boiling temperature)
        _Tb = 204.359 * np.log(np.matmul(self.Nij, Tbk))  # K
        self.Tb = u.Quantity(_Tb, "K")

        # T_m (melting temperature)
        _Tm = 102.425 * np.log(np.matmul(self.Nij, Tmk))  # K
        self.Tm = u.Quantity(_Tm, "K")

        # H_f (enthalpy of formation)
        _Hf = 10.835 + np.matmul(self.Nij, hfk)  # kJ/mol
        self.Hf = u.Quantity(_Hf, "kJ/mol").to("J/mol")

        # G_f (Gibbs free energy)
        _Gf = -14.828 + np.matmul(self.Nij, gfk)  # kJ/mol
        self.Gf = u.Quantity(_Gf, "kJ/mol").to("J/mol")

        # H_v,stp (enthalpy of vaporization at 298 K)
        _Hv_stp = 6.829 + (np.matmul(self.Nij, hvk))  # kJ/mol
        self.Hv_stp = u.Quantity(_Hv_stp, "kJ/mol").to("J/mol")

        # omega (accentric factor)
        _omega = 0.4085 * np.log(np.matmul(self.Nij, wk) + 1.1507) ** (1.0 / 0.5050)
        self.omega = u.Quantity(_omega, "dimensionless")

        # V_m (molar liquid volume at 298 K)
        _Vm_stp = 0.01211 + np.matmul(self.Nij, Vmk)  # m^3/kmol
        self.Vm_stp = u.Quantity(_Vm_stp, "m^3/kmol").to("m^3/mol")

        # C_p,stp (molar specific heat at 298 K)
        _Cp_stp = np.matmul(self.Nij, cpak) - 19.7779  # J/mol/K
        self.Cp_stp = u.Quantity(_Cp_stp, "J/(mol*K)")

        # Temperature corrections for C_p
        _Cp_B = np.matmul(self.Nij, cpbk)
        self.Cp_B = u.Quantity(_Cp_B, "J/(mol*K)")

        _Cp_C = np.matmul(self.Nij, cpck)
        self.Cp_C = u.Quantity(_Cp_C, "J/(mol*K)")

        # L_v,stp (latent heat of vaporization at 298 K)
        self.Lv_stp = (self.Hv_stp / self.MW).to("J/kg")  # J/kg

        # Lennard-Jones parameters for diffusion calculations (Tee et al. 1966)
        _epsilonByKB = (0.7915 + 0.1693 * self.omega) * self.Tc  # K
        self.epsilonByKB = u.Quantity(_epsilonByKB, "K")

        _sigma = (2.3551 - 0.0874 * self.omega) * (self.Tc / self.Pc.to("atm")) ** (
            1.0 / 3
        )  # Angstroms
        self.sigma = u.Quantity(ustrip(_sigma), "Angstrom").to("m")

    # -------------------------------------------------------------------------
    # Member functions
    # -------------------------------------------------------------------------
    def mean_molecular_weight(
        self, Yi: FloatArrayLike, *, unit: str = "kg/mol"
    ) -> u.Quantity:
        """
        Calculate the mean molecular weight of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: FloatArrayLike
        :param unit: Unit of the mean molecular weight (e.g., "kg/mol").
        :type unit: str
        :return: Mean molecular weight of the mixture in the specified unit.
        :rtype: u.Quantity
        """
        if np.sum(Yi) != 0:
            Mbar = 1 / np.sum(Yi / self.MW)  # mean molar weight of the mixture
        else:
            Mbar = u.Quantity(0.0, unit)

        return Mbar.to(unit)

    def mass2Y(self, mass: FloatArrayLike) -> npt.NDArray[np.float64]:
        """
        Calculate the mass fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: FloatArrayLike
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: npt.NDArray[np.float64]
        """
        # Normalize to get group mole fractions
        total_mass = np.array(np.sum(mass))
        if total_mass != 0:
            Yi = mass / total_mass
        else:
            Yi = np.zeros_like(self.MW, dtype=np.float64)

        return np.array(Yi, dtype=np.float64)

    def mass2X(self, mass: FloatArrayLike) -> npt.NDArray[np.float64]:
        """
        Calculate the mole fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: FloatArrayLike
        :return: Mole fractions of the compounds (shape: num_compounds,).
        :rtype: npt.NDArray[np.float64]
        """
        # Calculate the number of moles for each compound
        # NOTE: `mass` is not tracking units -- does not guarantee correct unit handling
        num_mole = np.array(mass / self.MW)

        # Normalize to get group mole fractions
        total_moles = np.sum(num_mole)
        if total_moles != 0:
            Xi = num_mole / total_moles
        else:
            Xi = np.zeros_like(self.MW, dtype=np.float64)

        return np.array(Xi, dtype=np.float64)

    def X2Y(self, Xi: FloatArrayLike) -> npt.NDArray[np.float64]:
        """
        Calculate the mass fractions from the mole fractions of each component.

        :param Xi: Mole fractions of each compound.
        :type Xi: FloatArrayLike
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: npt.NDArray[np.float64]
        """
        # Calculate the mass for each compound
        mass = np.array(Xi * self.MW)

        # Normalize to get group mass fractions
        total_mass = np.sum(mass)
        if total_mass != 0:
            Yi = mass / total_mass
        else:
            Yi = np.zeros_like(self.MW, dtype=np.float64)

        return np.array(Yi, dtype=np.float64)

    def Y2X(self, Yi: FloatArrayLike) -> npt.NDArray[np.float64]:
        """
        Calculate the mole fractions from the mass fractions of each component.

        :param Yi: Mass fractions of each compound.
        :type Yi: FloatArrayLike
        :return: Mole fractions of the compounds (shape: num_compounds,).
        :rtype: npt.NDArray[np.float64]
        """
        Mbar = self.mean_molecular_weight(Yi)
        if np.sum(Yi) != 0:
            Xi = Mbar * Yi / self.MW
        else:
            Xi = np.zeros_like(self.MW, dtype=np.float64)

        return np.array(Xi, dtype=np.float64)

    def density(
        self, T: u.Quantity, *, unit: str = "kg/m^3", comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Calculate the density of each component at temperature T.

        :param T: Temperature to calculate the density at.
        :type T: u.Quantity
        :param unit: Unit of the returned density.
        :type unit: str, optional
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Density of each compound in `unit`.
        :rtype: u.Quantity
        """
        T = convert_temperature(T, "K")
        rho = self.MW / self.molar_liquid_vol(T)
        if comp_idx is not None:
            rho = rho[comp_idx]
        return rho.to(unit)

    def viscosity_kinematic(
        self, T: u.Quantity, *, unit: str = "m^2/s", comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Calculate the viscosity using Dutt's equation.

        :meta private: This uses Dutt's equation (4.23) from "Viscosity of Liquids".
        :meta private: The equation predicts viscosity in mm^2/s and is converted to SI units.

        :param T: Temperature to calculate the kinematic viscosity at.
        :type T: u.Quantity
        :param unit: Unit of the returned kinematic viscosity.
        :type unit: str, optional
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Kinematic viscosity of each component in `unit`.
        :rtype: u.Quantity
        """
        T = convert_temperature(T, "Celsius")
        Tb = convert_temperature(self.Tb, "Celsius")

        # RHS of Dutt's equation (4.23) in Viscosity of Liquids (returns mm^2/s)
        # NOTE: Stripping units in the calculation then reattaching after exponential
        num = 442.78 + 1.6452 * ustrip(Tb)
        denom = ustrip(T) + 239 - 0.19 * ustrip(Tb)
        nu_i = u.Quantity(np.exp(-3.0171 + num / denom), "mm^2/s")

        if comp_idx is not None:
            nu_i = nu_i[comp_idx]

        return nu_i.to(unit)

    def viscosity_dynamic(
        self, T: u.Quantity, *, unit: str = "Pa*s", comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Calculate liquid dynamic viscosity based on droplet temperature and density.

        :meta private: Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

        :param T: Temperature to calculate the dynamic viscosity at.
        :type T: u.Quantity
        :param unit: Unit of the returned dynamic viscosity.
        :type unit: str, optional
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Dynamic viscosity in `unit`.
        :rtype: u.Quantity
        """
        nu_i = self.viscosity_kinematic(T, comp_idx=comp_idx)  # m^2/s
        rho_i = self.density(T, comp_idx=comp_idx)  # kg/m^3
        mu_i = nu_i * rho_i  # Pa*s
        return mu_i.to(unit)

    def Cp(
        self, T: u.Quantity, *, unit: str = "J/(mol*K)", comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Compute molar specific heat capacity at a given temperature.

        :param T: Temperature to calculate the molar specific heat capacity at.
        :type T: u.Quantity
        :param unit: Unit of the returned molar specific heat capacity.
        :type unit: str, optional
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar specific heat capacity in `unit`.
        :rtype: u.Quantity
        """
        # NOTE: Units are stripped for the calculation and reattached after the computation
        theta = (ustrip(T.to("K")) - 298) / 700

        Cp_stp = ustrip(self.Cp_stp.to(unit))
        Cp_B = ustrip(self.Cp_B.to(unit))
        Cp_C = ustrip(self.Cp_C.to(unit))

        cp = Cp_stp + Cp_B * theta + Cp_C * theta**2
        cp = u.Quantity(cp, "J/(mol*K)")

        if comp_idx is not None:
            cp = cp[comp_idx]

        return u.Quantity(cp, unit)

    def Cl(
        self, T: u.Quantity, *, unit: str = "J/(kg*K)", comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Compute liquid mass specific heat capacity at a given temperature.

        :param T: Temperature to calculate the liquid mass specific heat capacity at.
        :type T: u.Quantity
        :param unit: Unit of the returned mass specific heat capacity.
        :type unit: str, optional
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Mass specific heat capacity in `unit`.
        :rtype: u.Quantity
        """
        Cl = self.Cp(T) / self.MW

        if comp_idx is not None:
            Cl = Cl[comp_idx]

        return Cl.to(unit)

    def psat(
        self,
        T: u.Quantity,
        *,
        unit: str = "Pa",
        comp_idx: int | None = None,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> u.Quantity:
        """
        Compute saturated vapor pressure at a given temperature.

        :meta private: Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

        :param T: Temperature in Kelvin.
        :type T: float
        :param unit: Unit of the returned saturated vapor pressure.
        :type unit: str, optional
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Saturated vapor pressure in `unit`.
        :rtype: u.Quantity
        """
        # NOTE: Units are stripped for the calculation and reattached after the computation
        Tr = ustrip(T.to("K") / self.Tc)
        Pc = ustrip(self.Pc.to("Pa"))
        omega = ustrip(self.omega)

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

        elif (
            correlation.casefold() == "Lee-Kesler".casefold()
        ):  # Default correlation is Lee-Kesler
            f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
            f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
            rhs = np.exp(f0 + omega * f1)

        else:
            raise ValueError(f"Unrecognized correlation method: {correlation}")

        psat = u.Quantity(Pc * rhs, "Pa")

        if comp_idx is not None:
            psat = psat[comp_idx]

        return psat.to(unit)

    def psat_antoine_coeffs(
        self,
        Tvals: u.Quantity | None = None,
        *,
        unit: str = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Estimate Antoine coefficients for vapor pressure of an individual compound.

        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, Tb_i]).
        :type Tvals: np.ndarray, optional
        :param unit: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type unit: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D for the Antoine equation.
        :rtype: tuple of 4 np.ndarrays
        """

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, Tb_i] for each compound.")
            # Initialize as zeros for now, calculated for each compound later
            T = np.zeros(20) * u.K
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        T = convert_temperature(T, "K")
        Tb = ustrip(self.Tb.to("K"))

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(T, A, B, C):
            """Antoine equation for vapor pressure."""
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = ustrip((1 * u.Pa).to(unit))

        # Fit Antoine coefficients for each compound
        A = np.zeros(self.num_compounds)
        B = np.zeros(self.num_compounds)
        C = np.zeros(self.num_compounds)
        for i in range(self.num_compounds):
            # Update T if not specified
            if Tvals is None:
                T = np.linspace(273.15, Tb[i], 20) * u.K
            Pvals = np.zeros(len(T))
            for k in range(len(T)):
                Pval = 1 / D * self.psat(T[k], correlation=correlation)[i]
                Pvals[k] = ustrip(Pval)

            logP = np.log10(Pvals)
            popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])
            A[i], B[i], C[i] = popt

        return A, B, C, np.repeat(D, self.num_compounds)

    def molar_liquid_vol(
        self, T: u.Quantity, *, unit: str = "m^3/mol", comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Compute molar liquid volume with temperature correction.

        :param T: Temperature to evaluate the molar liquid volume at.
        :type T: u.Quantity
        :param unit: Unit to return the molar liquid volume in.
        :type unit: str, optional
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar liquid volume in specified unit.
        :rtype: u.Quantity
        """

        Tstp = 298.0 * u.Kelvin

        Tc = self.Tc
        omega = self.omega
        Vm_stp = self.Vm_stp

        x = -((1 - (Tstp / Tc)) ** (2.0 / 7.0))
        y = ((1 - (T / Tc)) ** (2.0 / 7.0)) + x
        phi = np.where(T > Tc, x, y)

        z = 0.29056 - 0.08775 * omega
        Vmi = Vm_stp * np.power(z, phi)

        if comp_idx is not None:
            Vmi = Vmi[comp_idx]

        return Vmi.to(unit)

    def latent_heat_vaporization(
        self, T: u.Quantity, *, unit: str = "J/kg", comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Calculate latent heat of vaporization adjusted for temperature.

        :param T: Temperature to evaluate latent heat of vaporization at.
        :type T: u.Quantity
        :param unit: Unit to return the latent heat of vaporization in.
        :type unit: str, optional
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Latent heat of vaporization in the specified unit.
        :rtype: u.Quantity
        """
        T = convert_temperature(T, "K")
        Tc = convert_temperature(self.Tc, "K")
        Tb = convert_temperature(self.Tb, "K")
        Lv_stp = self.Lv_stp

        # Reduced temperatures
        Tr = T / Tc
        Trb = Tb / Tc

        x = 0.0
        y = Lv_stp * (((1.0 - Tr) / (1.0 - Trb)) ** 0.38)
        Lvi = u.Quantity(np.where(T > Tc, x, y), "J/kg")

        if comp_idx is not None:
            Lvi = Lvi[comp_idx]

        return Lvi.to(unit)

    def diffusion_coeff(
        self,
        p: u.Quantity,
        T: u.Quantity,
        *,
        unit: str = "m^2/s",
        sigma_gas: u.Quantity = 3.62e-10 * u.m,
        epsilonByKB_gas: u.Quantity = 97.0 * u.K,
        MW_gas: u.Quantity = 28.97e-3 * u.kg / u.mol,
        correlation: Literal["Wilke", "Tee"] = "Tee",
    ) -> u.Quantity:
        """
        Compute diffusion coefficients using Lennard-Jones parameters.

        :meta private: Uses Wilke and Lee method (Poling, equation 11-4.1).
        :meta private: Ambient gas defaults to air parameters.

        :param p: Pressure to evaluate diffusion coefficient at.
        :type p: u.Quantity
        :param T: Temperature to evaluate diffusion coefficient at.
        :type T: u.Quantity
        :param unit: Unit to return the diffusion coefficient in.
        :type unit: str, optional
        :param sigma_gas: Collision diameter of the ambient gas.
        :type sigma_gas: u.Quantity, optional
        :param epsilonByKB_gas: Well depth over Boltzmann constant of the ambient gas.
        :type epsilonByKB_gas: u.Quantity, optional
        :param MW_gas: Mean molecular weight of ambient gas.
        :type MW_gas: u.Quantity, optional
        :param correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
        :type correlation: str, optional
        :return: Diffusion coefficient.
        :rtype: np.ndarray
        """
        # Method of Tee for calculating liquid sigma and epsilon
        if correlation.casefold() == "Tee".casefold():
            sigma_i = self.sigma.to("Angstrom")
            epsilonByKB_i = self.epsilonByKB  # K
        elif correlation.casefold() == "Wilke".casefold():
            # Method of Wilke & Lee calculating liquid sigma and epsilon
            Vmb_i = u.Quantity(np.zeros(self.num_compounds), "cm^3/mol")
            for n in range(self.num_compounds):
                Vmb_i[n] = self.molar_liquid_vol(self.Tb[n])[n].to("cm^3/mol")
            sigma_i = 1.18 * Vmb_i ** (1 / 3)  # Angstroms, Poling (11-4.2)
            sigma_i = u.Quantity(ustrip(sigma_i), "Angstrom")
            epsilonByKB_i = 1.15 * self.Tb.to("K")  # K , Poling (11-4.3)
        else:
            raise ValueError(f"Unrecognized correlation method: {correlation}")

        # Compute binary sigma and epsilon
        sigmaAB_i = (sigma_gas + sigma_i) / 2  # Angstroms, Poling (11-3.5)
        epsilonAB_byKB_i = (
            epsilonByKB_gas * epsilonByKB_i
        ) ** 0.5  # K, Poling (11-3.4)

        # Dimensionless collision integral for diffusion: Poling (11-3.6)
        Tstar_i = T / epsilonAB_byKB_i  # [1]
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

        # Convert molecular weights from kg/mol to g/mol then calculate M_AB
        M_AB_i = (
            2 * (self.MW * MW_gas) / (self.MW + MW_gas)
        )  # g/mol, see Poling (11-3.1)

        # NOTE: Units are stripped here then reattached after the calculation
        # NOTE: Declaring units within ustrip ensures they are in expected units
        _M_AB_i = ustrip(M_AB_i.to("g/mol"))
        _T = ustrip(T.to("K"))
        _p = ustrip(p.to("bar"))
        _sigmaAB_i = ustrip(sigmaAB_i.to("Angstrom"))
        _omegaD_i = ustrip(omegaD_i.to("dimensionless"))

        # Binary diffusion coefficients, Poling (11-4.1)
        D_AB_i = (
            1e-3
            * (3.03 - 0.98 / (_M_AB_i**0.5))
            * (_T**1.5)
            / (_p * _M_AB_i**0.5 * _sigmaAB_i**2 * _omegaD_i)
        )  # cm^2/s
        # Reattach units to D_AB_i
        D_AB_i = u.Quantity(D_AB_i, "cm^2/s")

        return D_AB_i.to(unit)

    def surface_tension(
        self,
        T: u.Quantity,
        *,
        unit: str = "N/m",
        comp_idx: int | None = None,
        correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
    ) -> u.Quantity:
        """
        Calculate surface tension of each compound at a given temperature.

        :meta private: Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

        :param T: Temperature to evaluate surface tension at.
        :type T: float
        :param unit: Unit to return the surface tension in.
        :type unit: str, optional
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Brock-Bird" or "Pitzer").
        :type correlation: Literal["Brock-Bird", "Pitzer"], optional
        :return: Surface tension in the specified unit.
        :rtype: u.Quantity
        """
        T_ = ustrip(convert_temperature(T, "K"))
        Tc = ustrip(self.Tc.to("Kelvin"))
        Pc = ustrip(self.Pc.to("bar"))
        Tb = ustrip(self.Tb.to("Kelvin"))
        omega = ustrip(self.omega)

        Tr = T_ / Tc

        if correlation.casefold() == "Brock-Bird".casefold():
            Tbr = Tb / Tc
            Q = 0.1196 * (1.0 + (Tbr * np.log(Pc / 1.01325)) / (1.0 - Tbr)) - 0.279

        elif correlation.casefold() == "Pitzer".casefold():
            Q = (
                (1.86 + 1.18 * omega)
                / 19.05
                * (((3.75 + 0.91 * omega) / (0.291 - 0.08 * omega)) ** (2.0 / 3.0))
            )
        else:
            raise ValueError(f"Unrecognized correlation method: {correlation}")

        st = Pc ** (2.0 / 3.0) * Tc ** (1.0 / 3.0) * Q * (1 - Tr) ** (11.0 / 9.0)
        st = u.Quantity(st, "dyn/cm")

        if comp_idx is not None:
            st = st[comp_idx]

        return u.Quantity(st).to(unit)

    def thermal_conductivity(
        self, T: u.Quantity, *, unit: str = "W/(m*K)", comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Calculate thermal conductivity at a given temperature.

        :meta private: Uses Latini et al. method (Poling equation 10-9.1).

        :param T: Temperature to evaluate thermal conductivity at.
        :type T: u.Quantity
        :param unit: Unit to return the thermal conductivity in.
        :type unit: str, optional
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Thermal conductivity in the specified unit.
        :rtype: u.Quantity
        """
        T_ = ustrip(convert_temperature(T, "K"))
        MW = ustrip(self.MW.to("kg/mol"))
        Tc = ustrip(self.Tc.to("Kelvin"))
        Tb = ustrip(self.Tb.to("Kelvin"))
        fam = self.fam

        Astar = 0.00350 + np.zeros(self.num_compounds)
        alpha = 1.2
        beta = 0.5 + np.zeros(self.num_compounds)
        gamma = 0.167
        MW_beta = MW * 1e3  # convert from kg/mol to g/mol
        Tr = T_ / Tc

        for i in range(len(Tc)):
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

        A = Astar * Tb**alpha / (MW_beta * Tc**gamma)
        tc = A * (1 - Tr) ** (0.38) / (Tr ** (1 / 6))
        tc = u.Quantity(tc, "W/(m*K)")

        if comp_idx is not None:
            tc = tc[comp_idx]

        return tc.to(unit)

    # --- Mixture functions ---
    def mixture_density(
        self, Yi: FloatArrayLike, T: u.Quantity, *, unit: str = "kg/m^3"
    ) -> u.Quantity:
        """
        Calculate mixture density at a given temperature.

        :param Yi: Mass fractions of each compound.
        :type Yi: FloatArrayLike
        :param T: Temperature to calculate the mixture density at.
        :type T: u.Quantity
        :param unit: Unit to return the mixture density in.
        :type unit: str, optional
        :return: Mixture density in the specified unit.
        :rtype: u.Quantity
        """
        return Yi @ self.density(T).to(unit)

    def mixture_kinematic_viscosity(
        self,
        Yi: FloatArrayLike,
        T: u.Quantity,
        *,
        unit: str = "m^2/s",
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> u.Quantity:
        """
        Calculate kinematic viscosity of the mixture.

        :meta private: Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature to calculate the mixture kinematic viscosity at.
        :type T: u.Quantity
        :param unit: Unit to return the mixture kinematic viscosity in.
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture kinematic viscosity in the specified unit.
        :rtype: u.Quantity
        """
        nu_i = ustrip(self.viscosity_kinematic(T).to("m^2/s"))
        # Calculate mole fractions for each species
        Xi = self.Y2X(Yi)

        if correlation.casefold() == "Arrhenius".casefold():
            # Arrhenius mixing correlation
            nu = np.exp(np.sum(Xi * np.log(nu_i)))
        elif correlation.casefold() == "Kendall-Monroe".casefold():
            # Default: Kendall-Monroe mixing correlation
            nu = np.sum(Xi * (nu_i ** (1.0 / 3.0))) ** (3.0)
        else:
            raise ValueError(f"Unsupported correlation method: {correlation}")

        return u.Quantity(nu, "m^2/s").to(unit)

    def mixture_dynamic_viscosity(
        self,
        Yi: FloatArrayLike,
        T: u.Quantity,
        *,
        unit: str = "Pa*s",
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> u.Quantity:
        """
        Calculate dynamic viscosity of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature to calculate the mixture dynamic viscosity at.
        :type T: u.Quantity
        :param unit: Unit to return the mixture dynamic viscosity in.
        :type unit: str, optional
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture dynamic viscosity in the specified unit.
        :rtype: u.Quantity
        """
        nu = self.mixture_kinematic_viscosity(Yi, T, correlation=correlation)
        rho = self.mixture_density(Yi, T)

        return (rho * nu).to(unit)

    def mixture_vapor_pressure(
        self,
        Yi: FloatArrayLike,
        T: u.Quantity,
        *,
        unit: str = "Pa",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> u.Quantity:
        """
        Calculate vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature to calculate the mixture vapor pressure at.
        :type T: u.Quantity
        :param unit: Unit to return the mixture vapor pressure in.
        :type unit: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Mixture vapor pressure in the specified unit.
        :rtype: u.Quantity
        """
        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Saturated vapor pressure for each compound (Pa)
        p_sati = self.psat(T, correlation=correlation)

        # Mixture vapor pressure via Raoult's law
        p_v = u.Quantity(p_sati.to("Pa") @ Xi, "Pa")

        return p_v.to(unit)

    def mixture_vapor_pressure_antoine_coeffs(
        self,
        Yi: FloatArrayLike,
        *,
        Tvals: u.Quantity | None = None,
        unit: str = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[float, float, float, float]:
        """
        Estimate Antoine coefficients for vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: FloatArrayLike
        :param Tvals: Temperature range or nodes for Antoine fit (default [273.15 K, min(Tb)]).
        :type Tvals: u.Quantity, optional
        :param unit: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type unit: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: tuple[float, float, float, float]
        """
        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, min(Tb_mix)] for mixture.")
            Xi = self.Y2X(Yi)
            Tb = mixing_rule(self.Tb, Xi).to("K")
            T = np.linspace(273.15 * u.K, np.min(Tb), 20)
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        T = convert_temperature(T, "K")

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
        D = ustrip((1 * u.Pa).to(unit))

        Pvals = np.zeros(len(T))
        for k in range(len(T)):
            Pval = self.mixture_vapor_pressure(Yi, T[k], correlation=correlation) / D
            Pvals[k] = ustrip(Pval)

        logP = np.log10(Pvals)
        popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])  # initial guess
        A, B, C = popt

        return float(A), float(B), float(C), float(D)

    def mixture_surface_tension(
        self,
        Yi: FloatArrayLike,
        T: u.Quantity,
        *,
        unit: str = "N/m",
        correlation: Literal["Pitzer", "Brock-Bird"] = "Brock-Bird",
    ) -> u.Quantity:
        """
        Calculate surface tension of the mixture.

        :meta private: Uses arithmetic pseudo-property method recommended by Hugill and van Welsenes (1986).

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature to calculate the surface tension at.
        :type T: u.Quantity
        :param correlation: Correlation method ("Pitzer" or "Brock-Bird").
        :type correlation: str, optional
        :return: Mixture surface tension in the specified unit.
        :rtype: u.Quantity in the specified unit
        """
        # Mole fraction for each compound
        Xi = self.Y2X(Yi)
        sti = self.surface_tension(T, correlation=correlation).to("N/m")
        # Mixture surface tension via arithmetic mean, Poling (12-5.2)
        return mixing_rule(sti, Xi, pseudo_prop="arithmetic").to(unit)

    def mixture_thermal_conductivity(
        self, Yi: FloatArrayLike, T: u.Quantity, *, unit: str = "W/(m*K)"
    ) -> u.Quantity:
        """
        Calculate thermal conductivity of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature to calculate the thermal conductivity at.
        :type T: u.Quantity
        :param unit: Unit of the thermal conductivity.
        :type unit: str, optional
        :return: Thermal conductivity in the specified unit.
        :rtype: u.Quantity in the specified unit
        """
        tc = self.thermal_conductivity(T)
        tc_mix = np.sum(Yi * tc.to(unit) ** (-2)) ** (-0.5)
        return u.Quantity(tc_mix, unit)


__all__ = ["fuel"]
