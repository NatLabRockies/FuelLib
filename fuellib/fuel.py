"""Fuel class for Group Contribution Method calculations."""

from __future__ import annotations
import os
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ._data_locator import (
    get_fueldata_decomp_dir,
    get_fueldata_dir,
    get_fueldata_gc_dir,
    get_fueldata_props_dir,
    get_gcmtable_dir,
    get_metadata_decomp_name,
)
from .constants import EpsilonByKB_gas, MW_gas, Sigma_gas
from .utility import mixing_rule
from .utils import Units, types


class Fuel:
    """Class for handling group contribution calculations of thermodynamic and mixture properties."""

    def __init__(
        self, name: str, decompName: str | None = None, fuelDataDir: str | None = None
    ) -> None:
        """Initialize Fuel object and pre-compute GCM properties.

        Args:
            name: Name of the mixture as it appears in its gcData file.
            decompName: Name of the groupDecomposition file if different from name. Defaults to None.
            fuelDataDir: Directory where the fuel data is stored. If None, uses built-in embedded data.

        Raises:
            ValueError: If a GCM property cannot be found.
        """
        self.name: str = name
        """Name of the fuel/mixture."""
        if decompName is None:
            # Try to get decomposition name from metadata
            decompName: str = get_metadata_decomp_name(name, fuelDataDir)
            """Name of the group decomposition file."""

        # Determine and set data directories for this fuel instance
        if fuelDataDir is None:
            # Use built-in embedded data
            self.fuelDataDir: str = get_fueldata_dir()
            """Directory containing the fuel data."""
            self.fuelDataGcDir: str = get_fueldata_gc_dir()
            """Directory containing the gas chromatography data."""
            self.fuelDataDecompDir: str = get_fueldata_decomp_dir()
            """Directory containing the group decomposition data."""
            self.fuelDataPropsDir: str = get_fueldata_props_dir()
            """Directory containing the fuel properties data."""
        else:
            # Validate and use custom fuel directory
            from ._data_locator import (
                _get_props_dir_for_fueldata,
                _validate_fuel_data_dir,
            )

            _validate_fuel_data_dir(fuelDataDir)
            self.fuelDataDir: str = fuelDataDir
            """Directory containing the fuel data."""
            self.fuelDataGcDir: str = os.path.join(fuelDataDir, "gcData")
            """Directory containing the gas chromatography data."""
            self.fuelDataDecompDir: str = os.path.join(
                fuelDataDir, "groupDecompositionData"
            )
            """Directory containing the group decomposition data."""
            self.fuelDataPropsDir: str = _get_props_dir_for_fueldata(fuelDataDir)
            """Directory containing the fuel properties data."""

        # Get GCM table directory (always from built-in data)
        gcmtable_dir = get_gcmtable_dir()

        self.groupDecompFile: str = os.path.join(
            self.fuelDataDecompDir, f"{decompName}.csv"
        )
        """File containing the group decomposition data for this fuel."""
        self.gcxgcFile: str = os.path.join(self.fuelDataGcDir, f"{name}_init.csv")
        """File containing the GCxGC compositional data for this fuel."""
        self.gcmTableFile: str = os.path.join(gcmtable_dir, "gcmTable.csv")
        """File containing the GCM table data."""

        # Read functional group data for mixture (num_compounds,num_groups)
        df_Nij = pd.read_csv(self.groupDecompFile)
        self.Nij: types.Array2D = df_Nij.iloc[:, 1:].to_numpy()
        """Array containing the group decomposition data for each compound."""
        self.num_compounds: int = self.Nij.shape[0]
        """Number of compounds in the fuel mixture."""
        self.num_groups: int = self.Nij.shape[1]
        """Number of functional groups considered in the decomposition."""

        self.fam: types.Array1D = np.zeros(self.num_compounds, dtype=int)
        """Hydrocarbon family codes for thermal conductivity.

        ==== ==================
        Code Hydrocarbon Family
        ==== ==================
        0    saturated
        1    aromatics
        2    cycloparaffins
        3    olefins
        ==== ==================

        """

        # Classify hydrocarbon by type (n-alkane, iso-alkane, cyclo-alkane, aromatic)
        # Based on group decompositions from Constantinou-Gani method
        self.hc_type: types.Array1D = np.array([""] * self.num_compounds, dtype=object)
        """Hydrocarbon types for each compound:

        * "n-alkane"
        * "iso-alkane"
        * "alkene"
        * "cyclo-alkane"
        * "aromatic"
        """

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

        self.nC: types.Array1D = np.zeros(self.num_compounds, dtype=float)
        """Number of carbon atoms in each compound."""
        self.nH: types.Array1D = np.zeros(self.num_compounds, dtype=float)
        """Number of hydrogen atoms in each compound."""
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

        self.compounds: list[str] = [
            compound.strip() for compound in df_gcxgc["Compound"].to_list()
        ]
        """List of compound names."""

        # Load molecular formulas if available
        if "Formula" in df_gcxgc.columns:
            self.formulas: types.Array1D | None = np.array([
                formula.strip() if pd.notna(formula) else None
                for formula in df_gcxgc["Formula"].to_list()
            ])
            """Molecular formulas of the fuel components, if available."""
        else:
            self.formulas = None

        if "PelePhysics Key" in df_gcxgc.columns:
            self.pelephysics_keys: types.Array1D | None = np.array([
                key.strip() for key in df_gcxgc["PelePhysics Key"].to_list()
            ])
            """PelePhysics keys for the fuel components, if available."""
        else:
            self.pelephysics_keys = None

        self.Y_0: types.Array1D = (
            df_gcxgc["Weight %"].to_numpy().flatten().astype(float)
        )
        """Initial mass fractions of the fuel components."""
        self.Y_0 /= np.sum(self.Y_0)
        self.Y_0: types.Quantity1D = Units.Quantity(self.Y_0, "dimensionless")

        # Make sure mixture data is consistent:
        _N_g1 = 78
        _N_g2 = 43
        if self.num_groups < _N_g1:
            raise ValueError(
                f"Insufficient mixture description:\n"
                f"The number of columns in {self.groupDecompFile} is less than "
                f"the required number of first-order groups (N_g1 = {_N_g1})."
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

        def _get_row(property_name: str) -> types.Array1D:
            """Get property row from GCM table.

            Args:
                property_name: Name of the property to retrieve.

            Returns:
                Property values for all functional groups.

            Raises:
                ValueError: If property not found in GCM table.
            """
            row = df_table[df_table["Property"] == property_name]
            if row.empty:
                raise ValueError(f"Property '{property_name}' not found in GCM table.")
            return row.iloc[:, 1:].to_numpy().flatten()

        # Table data for functional groups (num_compounds,)
        _Tck = _get_row("tck")  # critical temperature (1)
        _Pck = _get_row("pck")  # critical pressure (bar)
        _Vck = _get_row("vck")  # critical volume (m^3/kmol)
        _Tbk = _get_row("tbk")  # boiling temperature (1)
        _Tmk = _get_row("tmk")  # melting point temperature (1)
        _hfk = _get_row("hfk")  # enthalpy of formation, (kJ/mol)
        _gfk = _get_row("gfk")  # Gibbs energy (kJ/mol)
        _hvk = _get_row("hvk")  # latent heat of vaporization (kJ/mol)
        _wk = _get_row("wk")  # accentric factor (1)
        _Vmk = _get_row("vmk")  # liquid molar volume fraction (m^3/kmol)
        _cpak = _get_row("CpAk")  # specific heat values (J/mol/K)
        _cpbk = _get_row("CpBk")  # specific heat values (J/mol/K)
        _cpck = _get_row("CpCk")  # specific heat values (J/mol/K)
        _mwk = _get_row("MW")  # molecular weights (g/mol)

        # --- Compute critical properties at standard temp (num_compounds,)
        # Molecular weights

        _mw = np.matmul(self.Nij, _mwk)
        self.MW: types.Quantity1D = Units.Quantity(_mw, "g/mol").to("kg/mol")
        """Molecular weights in kg/mol."""

        # T_c (critical temperature)
        _tc = 181.128 * np.log(np.matmul(self.Nij, _Tck))
        self.Tc: types.Quantity1D = Units.Quantity(_tc, "K")
        """Critical temperature in K."""

        # p_c (critical pressure)
        _pc = 1.3705 + (np.matmul(self.Nij, _Pck) + 0.10022) ** (-2)
        self.Pc: types.Quantity1D = Units.Quantity(_pc, "bar").to("Pa")
        """Critical pressure in Pa."""

        # V_c (critical volume)
        _vc = -0.00435 + np.matmul(self.Nij, _Vck)
        self.Vc: types.Quantity1D = Units.Quantity(_vc, "m^3/kmol").to("m^3/mol")
        """Critical volume in m^3/mol."""

        # T_b (boiling temperature)
        _tb = 204.359 * np.log(np.matmul(self.Nij, _Tbk))
        self.Tb: types.Quantity1D = Units.Quantity(_tb, "K")
        """Boiling temperature in K."""

        # T_m (melting temperature)
        _tm = 102.425 * np.log(np.matmul(self.Nij, _Tmk))
        self.Tm: types.Quantity1D = Units.Quantity(_tm, "K")
        """Melting temperature in K."""

        # H_f (enthalpy of formation)
        _hf = 10.835 + np.matmul(self.Nij, _hfk)
        self.Hf: types.Quantity1D = Units.Quantity(_hf, "kJ/mol").to("J/mol")
        """Enthalpy of formation in J/mol."""

        # G_f (Gibbs free energy)
        _gf = -14.828 + np.matmul(self.Nij, _gfk)
        self.Gf: types.Quantity1D = Units.Quantity(_gf, "kJ/mol").to("J/mol")
        """Gibbs free energy in J/mol."""

        # H_v,stp (enthalpy of vaporization at 298 K)
        _hv_stp = 6.829 + np.matmul(self.Nij, _hvk)
        self.Hv_stp: types.Quantity1D = Units.Quantity(_hv_stp, "kJ/mol").to("J/mol")
        """Enthalpy of vaporization at 298 K in J/mol."""

        # omega (accentric factor)
        _omega = 0.4085 * np.log(np.matmul(self.Nij, _wk) + 1.1507) ** (1.0 / 0.5050)
        self.omega: types.Quantity1D = Units.Quantity(_omega, "")
        """Accentric factor (dimensionless)."""

        # V_m (molar liquid volume at 298 K)
        _vm_stp = 0.01211 + np.matmul(self.Nij, _Vmk)
        self.Vm_stp: types.Quantity1D = Units.Quantity(_vm_stp, "m^3/kmol").to(
            "m^3/mol"
        )
        """Molar liquid volume at 298 K in m^3/mol."""

        # C_p,stp (molar specific heat at 298 K)
        _cp_stp = np.matmul(self.Nij, _cpak) - 19.7779
        self.Cp_stp: types.Quantity1D = Units.Quantity(_cp_stp, "J/(mol*K)")
        """Molar specific heat at 298 K in J/(mol*K)."""

        # Temperature corrections for C_p
        _cp_b = np.matmul(self.Nij, _cpbk)
        self.Cp_B: types.Quantity1D = Units.Quantity(_cp_b, "J/(mol*K)")
        """Temperature-corrected specific heat (B) in J/(mol*K)."""
        _cp_c = np.matmul(self.Nij, _cpck)
        self.Cp_C: types.Quantity1D = Units.Quantity(_cp_c, "J/(mol*K)")
        """Temperature-corrected specific heat (C) in J/(mol*K)."""

        # L_v,stp (latent heat of vaporization at 298 K)
        self.Lv_stp: types.Quantity1D = (self.Hv_stp / self.MW).to("J/kg")
        """Latent heat of vaporization at 298 K in J/kg."""

        # Lennard-Jones parameters for diffusion calculations (Tee et al. 1966)
        _lj_w = self.omega.magnitude
        _lj_tc = self.Tc.to("K").magnitude
        _lj_pc = self.Pc.to("atm").magnitude
        _epsilon_by_kb = (0.7915 + 0.1693 * _lj_w) * _lj_tc
        self.epsilonByKB: types.Quantity1D = Units.Quantity(_epsilon_by_kb, "K")
        """Lennard-Jones well depth over Boltzmann constant in K."""

        _sigma = (2.3551 - 0.0874 * _lj_w) * (_lj_tc / _lj_pc) ** (1.0 / 3)
        self.sigma: types.Quantity1D = Units.Quantity(_sigma, "angstrom").to("m")
        """Lennard-Jones collision diameter in m."""

    # -------------------------------------------------------------------------
    # Member functions
    # -------------------------------------------------------------------------
    def mean_molecular_weight(self, Yi: types.Quantity1D) -> types.Quantity0D:
        """Calculate the mean molecular weight of the mixture.

        Args:
            Yi: Mass fractions of each compound.

        Returns:
            Mean molecular weight of the mixture in kg/mol.
        """
        MW = self.MW.to("kg/mol")
        if np.sum(Yi) != 0:
            Mbar = 1 / np.sum(Yi / MW)
        else:
            Mbar = Units.Quantity(0.0, "kg/mol")

        return Mbar

    def mass2Y(self, mass: types.Quantity1D) -> types.Quantity1D:
        """Calculate the mass fractions from the mass of each component.

        Args:
            mass: Mass of each compound.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        # Normalize to get group mole fractions
        mass = mass.to("kg")
        total_mass = mass.magnitude.sum()
        if total_mass != 0:
            Yi = (mass / total_mass).magnitude
        else:
            Yi = np.zeros_like(self.MW.magnitude)

        return Units.Quantity(Yi, "dimensionless")

    def mass2X(self, mass: types.Quantity1D) -> types.Quantity1D:
        """Calculate the mole fractions from the mass of each component.

        Args:
            mass: Mass of each compound.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        mass = mass.to("kg")

        # Calculate the number of moles for each compound
        num_mole = mass / self.MW

        # Normalize to get group mole fractions
        total_moles = np.sum(num_mole)
        if total_moles != 0:
            Xi = (num_mole / total_moles).magnitude
        else:
            Xi = np.zeros_like(self.MW.magnitude)

        return Units.Quantity(Xi, "dimensionless")

    def X2Y(self, Xi: types.Quantity1D) -> types.Quantity1D:
        """Calculate the mass fractions from the mole fractions of each component.

        Args:
            Xi: Mole fractions of each compound.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        # Calculate the mass for each compound
        mass = self.MW * Xi
        # Normalize to get group mass fractions
        total_mass = np.sum(mass)
        Yi: types.Array1D = (
            (mass / total_mass).magnitude
            if total_mass != 0
            else np.zeros_like(self.MW.magnitude)
        )

        return Units.Quantity(Yi, "dimensionless")

    def Y2X(self, Yi: types.Quantity1D) -> types.Quantity1D:
        """Calculate the mole fractions from the mass fractions of each component.

        Args:
            Yi: Mass fractions of each compound.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        Mbar = self.mean_molecular_weight(Yi)
        if np.sum(Yi) != 0:
            Xi = (Mbar * Yi / self.MW).magnitude
        else:
            Xi = np.zeros_like(self.MW.magnitude)

        return Units.Quantity(Xi, "dimensionless")

    def density(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate the density of each component at temperature T.

        Args:
            T: Temperature of the mixture in Kelvin.
            comp_idx: Index of compound to calculate property for.

        Returns:
            Density of each compound in kg/m^3.
        """
        T = T.to("K")
        if comp_idx is None:
            MW = self.MW
            Vm = self.molar_liquid_vol(T)
        else:
            MW = self.MW[comp_idx]
            Vm = self.molar_liquid_vol(T, comp_idx=comp_idx)

        rho = (MW / Vm).to("kg/m^3")
        return rho

    def viscosity_kinematic(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate the viscosity using Dutt's equation.

        Uses Dutt's equation (4.23) from "Viscosity of Liquids". The equation
        predicts viscosity in mm^2/s and is converted to SI units.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.

        Returns:
            Viscosity of each component in m^2/s.
        """
        # Convert temperature to Celsius
        T: float = T.to("celsius").magnitude
        if comp_idx is None:
            Tb = self.Tb.to("celsius").magnitude
        else:
            Tb = self.Tb[comp_idx].to("celsius").magnitude

        # RHS of Dutt's equation (4.23) in Viscosity of Liquids
        rhs = -3.0171 + (442.78 + 1.6452 * Tb) / (T + 239 - 0.19 * Tb)
        nu_i = Units.Quantity(np.exp(rhs), "mm^2/s").to("m^2/s")

        return nu_i

    def viscosity_dynamic(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate liquid dynamic viscosity based on droplet temperature and density.

        Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.

        Returns:
            Dynamic viscosity in Pa*s.
        """
        nu_i = self.viscosity_kinematic(T, comp_idx=comp_idx)
        rho_i = self.density(T, comp_idx=comp_idx)
        mu_i = (nu_i * rho_i).to("Pa*s")
        return mu_i

    def Cp(self, T: types.Quantity0D, comp_idx: int | None = None) -> types.Quantity1D:
        """Compute molar specific heat capacity at a given temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.

        Returns:
            Molar specific heat capacity in J/mol/K.
        """
        T = T.to("K")
        theta = (T - Units.Quantity(298, "K")) / Units.Quantity(700, "K")
        if comp_idx is None:
            Cp_stp = self.Cp_stp
            Cp_B = self.Cp_B
            Cp_C = self.Cp_C
        else:
            Cp_stp = self.Cp_stp[comp_idx]
            Cp_B = self.Cp_B[comp_idx]
            Cp_C = self.Cp_C[comp_idx]

        cp = Cp_stp + Cp_B * theta + Cp_C * theta**2

        return cp.to("J/(mol*K)")

    def Cl(self, T: types.Quantity0D, comp_idx: int | None = None) -> types.Quantity1D:
        """Compute liquid mass specific heat capacity in J/kg/K at a given temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.

        Returns:
            Mass specific heat capacity in J/kg/K.
        """
        T = T.to("K")
        if comp_idx is None:
            MW = self.MW
        else:
            MW = self.MW[comp_idx]
        cp = self.Cp(T, comp_idx=comp_idx)
        return (cp / MW).to("J/(kg*K)")

    def psat(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> types.Quantity1D:
        """Compute saturated vapor pressure.

        Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

        Returns:
            Saturated vapor pressure in Pa.
        """
        T = T.to("K")
        if comp_idx is None:
            Tr = T / self.Tc
            Pc = self.Pc
            omega = self.omega
        else:
            Tr = T / self.Tc[comp_idx]
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

        return (Pc * rhs).to("Pa")

    def psat_antoine_coeffs(
        self,
        Tvals: types.Quantity1D | None = None,
        units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[types.Array1D, types.Array1D, types.Array1D, types.Array1D]:
        """Estimate Antoine coefficients for vapor pressure of an individual compound.

        Args:
            Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, Tb_i]).
            units: Units for pressure in fit ("mks", "cgs").
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

        Returns:
            Coefficients A, B, C, D for each compound.

        Raises:
            ValueError: If units or Tvals are invalid.
        """
        if units == "cgs":
            units = "dyne/cm^2"
        elif units == "mks":
            units = "Pa"
        else:
            raise ValueError("units must be either 'mks' or 'cgs'.")

        if Tvals is not None:
            Tvals = Tvals.to("K")

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, Tb_i] for each compound.")
            # Initialize as zeros for now, calculated for each compound later
            T = Units.Quantity(np.zeros(20), "K")
        elif len(Tvals) == 2:
            T_low = Tvals[0].magnitude
            T_high = Tvals[1].magnitude
            T = Units.Quantity(
                np.linspace(T_low, T_high, 20),
                "K",
            )
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(
            T: float | types.Array1D, A: float, B: float, C: float
        ) -> float | types.Array1D:
            """Antoine equation for vapor pressure.

            Args:
                T: Temperature.
                A: Antoine coefficient A.
                B: Antoine coefficient B.
                C: Antoine coefficient C.

            Returns:
                log10(pressure).
            """
            return A - B / (T + C)

        # Fit A, B, C against pressure in Pa (mks base) so the coefficients are
        # unit independent. "mks" (meter-kilogram-second) and "cgs"
        # D is the Pa-to-target-unit conversion factor, applied only when evaluating
        # psat(T) = D * 10**(A - B/(T + C)) in Pele.
        D = Units.Quantity(1, "Pa").to(units)

        # Fit Antoine coefficients for each compound
        A = np.zeros(self.num_compounds)
        B = np.zeros(self.num_compounds)
        C = np.zeros(self.num_compounds)
        for i in range(self.num_compounds):
            # Update T if not specified
            if Tvals is None:
                T = Units.Quantity(np.linspace(273.15, self.Tb[i].magnitude, 20), "K")
            T_magnitude = T.to("K").magnitude
            Pvals = np.zeros_like(T_magnitude)
            for k in range(len(T)):
                Pvals[k] = (
                    self.psat(T[k], correlation=correlation)[i].to("Pa").magnitude
                )

            logP = np.log10(Pvals)
            popt, _ = curve_fit(antoine_eq, T_magnitude, logP, p0=[1, 1e3, -1])
            A[i], B[i], C[i] = popt
        D = D.magnitude + np.zeros(self.num_compounds)  # make D an array
        return A, B, C, D

    def molar_liquid_vol(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Compute molar liquid volume with temperature correction.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.

        Returns:
            Molar liquid volume in m^3/mol.
        """
        Tstp = Units.Quantity(298, "K")
        T = T.to("K")
        if comp_idx is None:
            Tc = self.Tc.to("K")
            omega = self.omega
            Vm_stp = self.Vm_stp
        else:
            Tc = self.Tc[comp_idx : comp_idx + 1]
            omega = self.omega[comp_idx : comp_idx + 1]
            Vm_stp = self.Vm_stp[comp_idx : comp_idx + 1]
        phi = np.zeros_like(Tc.magnitude)
        for i in range(len(Tc)):
            if T > Tc[i]:
                phi[i] = -((1 - (Tstp / Tc[i])) ** (2.0 / 7.0))
            else:
                phi[i] = (1 - (T / Tc[i])) ** (2.0 / 7.0) - (1 - (Tstp / Tc[i])) ** (
                    2.0 / 7.0
                )
        z = 0.29056 - 0.08775 * omega
        Vmi = Vm_stp * z**phi
        if comp_idx is not None:
            Vmi = Vmi[0]
        return Vmi

    def latent_heat_vaporization(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate latent heat of vaporization adjusted for temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.

        Returns:
            Latent heat of vaporization in J/kg.
        """
        T = T.to("K")

        if comp_idx is None:
            Tc = self.Tc.to("K")
            Tb = self.Tb.to("K")
            Lv_stp = self.Lv_stp.to("J/kg")
        else:
            Tc = self.Tc[comp_idx : comp_idx + 1].to("K")
            Tb = self.Tb[comp_idx : comp_idx + 1].to("K")
            Lv_stp = self.Lv_stp[comp_idx : comp_idx + 1].to("J/kg")

        # Reduced temperatures
        Tr = T / Tc
        Trb = Tb / Tc

        Lvi = Units.Quantity(np.zeros_like(Tc.magnitude), "J/kg")
        for i in range(len(Tc)):
            if T > Tc[i]:
                Lvi.magnitude[i] = 0.0
            else:
                Lvi[i] = Lv_stp[i] * (((1.0 - Tr[i]) / (1.0 - Trb[i])) ** 0.38)

        if comp_idx is not None:
            Lvi = Lvi[0]
        return Lvi

    def diffusion_coeff(
        self,
        p: types.Quantity0D,
        T: types.Quantity0D,
        sigma_gas: types.Quantity0D = Sigma_gas,
        epsilonByKB_gas: types.Quantity0D = EpsilonByKB_gas,
        MW_gas: types.Quantity0D = MW_gas,
        correlation: Literal["Tee", "Wilke"] = "Tee",
    ) -> types.Quantity1D:
        """Compute diffusion coefficients using Lennard-Jones parameters.

        Uses Wilke and Lee method (Poling, equation 11-4.1). Ambient gas
        defaults to air parameters.

        Args:
            p: Pressure in Pa.
            T: Temperature to compute property.
            sigma_gas: Collision diameter in m.
            epsilonByKB_gas: Well depth over Boltzmann constant, in K.
            MW_gas: Mean molecular weight of ambient gas in kg/mol.
            correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").

        Returns:
            Diffusion coefficient.
        """
        p = p.to("bar")
        T = T.to("K")
        sigma_gas = sigma_gas.to("angstrom")
        epsilonByKB_gas = epsilonByKB_gas.to("K")
        MW_gas = MW_gas.to("g/mol")

        # Method of Tee for calculating liquid sigma and epsilon
        if correlation.casefold() == "Tee".casefold():
            sigma_i = self.sigma.to("angstrom").magnitude
            epsilonByKB_i = self.epsilonByKB.to("K").magnitude
        else:
            # Method of Wilke & Lee calculating liquid sigma and epsilon
            Vmb_i = np.zeros_like(self.Tb.magnitude)
            for n in range(self.num_compounds):
                Vmb_i[n] = self.molar_liquid_vol(self.Tb[n])[n].to("cm^3/mol").magnitude
            sigma_i = 1.18 * Vmb_i ** (1 / 3)  # Angstroms, Poling (11-4.2)
            epsilonByKB_i = 1.15 * self.Tb.to("K").magnitude  # K, Poling (11-4.3)

        # Compute binary sigma and epsilon
        sigma_gas: float = sigma_gas.magnitude
        sigmaAB_i = (sigma_gas + sigma_i) / 2  # Angstroms, Poling (11-3.5)
        epsilonAB_byKB_i = (
            epsilonByKB_gas.magnitude * epsilonByKB_i
        ) ** 0.5  # K, Poling (11-3.4)

        # Dimensionless collision integral for diffusion: Poling (11-3.6)
        T: float = T.magnitude
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
        MW_gas: float = MW_gas.magnitude
        MW_i = self.MW.to("g/mol").magnitude
        M_AB_i = 2 * (MW_i * MW_gas) / (MW_i + MW_gas)  # g/mol, see Poling (11-3.1)

        # Pressure is already in bar.
        p: float = p.magnitude

        # Binary diffusion coefficients, Poling (11-4.1)
        D_AB_i = (
            1e-3
            * (3.03 - 0.98 / (M_AB_i**0.5))
            * (T**1.5)
            / (p * M_AB_i**0.5 * sigmaAB_i**2 * omegaD_i)
        )  # cm^2/s
        return Units.Quantity(D_AB_i, "cm^2/s").to("m^2/s")

    def surface_tension(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
    ) -> types.Quantity1D:
        """Calculate surface tension of each compound at a given temperature.

        Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
            correlation: Correlation method ("Brock-Bird" or "Pitzer").

        Returns:
            Surface tension in N/m.
        """
        T = T.to("K")
        if comp_idx is None:
            Tc = self.Tc.to("K").magnitude
            Pc = self.Pc.to("Pa").magnitude
            Tb = self.Tb.to("K").magnitude
            omega = self.omega.magnitude
        else:
            Tc = np.array([self.Tc[comp_idx].to("K").magnitude])
            Pc = np.array([self.Pc[comp_idx].to("Pa").magnitude])
            Tb = np.array([self.Tb[comp_idx].to("K").magnitude])
            omega = np.array([self.omega[comp_idx].magnitude])
        Tr = T.magnitude / Tc
        Pc = Pc * 1e-5  # convert from Pa to bar

        if correlation.casefold() == "Brock-Bird".casefold():
            Tbr = Tb / Tc
            Q = 0.1196 * (1.0 + (Tbr * np.log(Pc / 1.01325)) / (1.0 - Tbr)) - 0.279
        else:
            w = omega
            Q = (
                (1.86 + 1.18 * w)
                / 19.05
                * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
            )

        st = Pc ** (2.0 / 3.0) * Tc ** (1.0 / 3.0) * Q * (1 - Tr) ** (11.0 / 9.0)

        st = Units.Quantity(st, "dyn/cm").to("N/m")
        if comp_idx is not None:
            st = st[0]

        return st

    def thermal_conductivity(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
    ) -> types.Quantity1D:
        """Calculate thermal conductivity at a given temperature.

        Uses Latini et al. method (Poling equation 10-9.1).

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.

        Returns:
            Thermal conductivity in W/m/K.
        """
        T = T.to("K")
        if comp_idx is None:
            MW = self.MW.to("kg/mol").magnitude
            Tc = self.Tc.to("K").magnitude
            Tb = self.Tb.to("K").magnitude
            fam = self.fam
        else:
            MW = np.array([self.MW[comp_idx].to("kg/mol").magnitude])
            Tc = np.array([self.Tc[comp_idx].to("K").magnitude])
            Tb = np.array([self.Tb[comp_idx].to("K").magnitude])
            fam = np.array([self.fam[comp_idx]])

        Astar = 0.00350 + np.zeros_like(Tc)
        alpha = 1.2
        beta = 0.5 + np.zeros_like(Tc)
        gamma = 0.167
        MW_beta = MW * 1e3  # convert from kg/mol to g/mol
        Tr = T.magnitude / Tc

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

        if comp_idx is not None:
            tc = tc[0]
        return Units.Quantity(tc, "W/(m*K)")

    # --- Mixture functions ---
    def mixture_density(
        self, Yi: types.Quantity1D, T: types.Quantity0D
    ) -> types.Quantity1D:
        """Calculate mixture density at a given temperature.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.

        Returns:
            Mixture density in kg/m^3.
        """
        T = T.to("K")
        MW = self.MW.to("kg/mol")
        Vmi = self.molar_liquid_vol(T).to("m^3/mol")

        # Calculate density (kg/m^3)
        rho = (Yi @ (MW / Vmi)).to("kg/m^3")

        return rho

    def mixture_kinematic_viscosity(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> types.Quantity0D:
        """Calculate kinematic viscosity of the mixture.

        Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.
            correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").

        Returns:
            Mixture kinematic viscosity in m^2/s.
        """
        T = T.to("K")
        nu_i = self.viscosity_kinematic(T).to("m^2/s").magnitude

        # Calculate mole fractions for each species
        Xi = self.Y2X(Yi).magnitude

        if correlation.casefold() == "Arrhenius".casefold():
            # Arrhenius mixing correlation
            nu = np.exp(np.sum(Xi * np.log(nu_i)))
        else:
            # Default: Kendall-Monroe mixing correlation
            nu = np.sum(Xi * (nu_i ** (1.0 / 3.0))) ** 3.0

        return Units.Quantity(nu, "m^2/s")

    def mixture_dynamic_viscosity(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> types.Quantity0D:
        """Calculate dynamic viscosity of the mixture.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.
            correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").

        Returns:
            Mixture dynamic viscosity in Pa*s.
        """
        T = T.to("K")
        nu = self.mixture_kinematic_viscosity(Yi, T, correlation=correlation)
        rho = self.mixture_density(Yi, T)

        return (rho * nu).to("Pa*s")

    def mixture_vapor_pressure(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> types.Quantity0D:
        """Calculate vapor pressure of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

        Returns:
            Mixture vapor pressure in Pa.
        """
        T = T.to("K")

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Saturated vapor pressure for each compound (Pa)
        p_sati = self.psat(T, correlation=correlation).to("Pa")

        # Mixture vapor pressure via Raoult's law
        p_v = p_sati @ Xi

        return p_v.to("Pa")

    def mixture_vapor_pressure_antoine_coeffs(
        self,
        Yi: types.Quantity1D,
        Tvals: types.Quantity1D | None = None,
        units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[float, float, float, float]:
        """Estimate Antoine coefficients for vapor pressure of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, min(Tb)]).
            units: Units for pressure in fit ("mks", "cgs").
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

        Returns:
            Coefficients A, B, C, D.

        Raises:
            ValueError: If units or Tvals are invalid.
        """
        if units == "cgs":
            units = "dyne/cm^2"
        elif units == "mks":
            units = "Pa"
        elif units not in ["dyne/cm^2", "Pa"]:
            raise ValueError("units must be either 'mks', 'cgs', 'dyne/cm^2', or 'Pa'.")

        if Tvals is not None:
            Tvals = Tvals.to("K")

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, min(Tb_mix)] for mixture.")
            X = self.Y2X(Yi).magnitude
            Tb = mixing_rule(self.Tb, X)
            T = Units.Quantity(
                np.linspace(273.15, np.min(Tb.to("K").magnitude), 20), "K"
            )
        elif len(Tvals) == 2:
            T = Units.Quantity(
                np.linspace(Tvals[0].magnitude, Tvals[1].magnitude, 20), "K"
            )
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(
            T: float | types.Array1D, A: float, B: float, C: float
        ) -> float | types.Array1D:
            """Antoine equation for vapor pressure.

            Args:
                T: Temperature.
                A: Antoine coefficient A.
                B: Antoine coefficient B.
                C: Antoine coefficient C.

            Returns:
                log10(pressure).
            """
            return A - B / (T + C)

        # Fit A, B, C against pressure in Pa (mks base) so the coefficients are
        # unit independent. "mks" (meter-kilogram-second) and "cgs"
        # D is the Pa-to-target-unit conversion factor, applied only when evaluating
        # psat(T) = D * 10**(A - B/(T + C)) in Pele.
        D = Units.Quantity(1, "Pa").to(units)

        T_magnitude = T.to("K").magnitude
        Pvals = np.zeros_like(T_magnitude)
        for k in range(len(T)):
            Pvals[k] = (
                self
                .mixture_vapor_pressure(Yi, T[k], correlation=correlation)
                .to(units)
                .magnitude
            )

        logP = np.log10(Pvals)
        popt, _ = curve_fit(antoine_eq, T_magnitude, logP, p0=[1, 1e3, -1])
        A, B, C = popt

        return A, B, C, D.magnitude

    def mixture_surface_tension(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Pitzer", "Brock-Bird"] = "Brock-Bird",
    ) -> types.Quantity0D:
        """Calculate surface tension of the mixture.

        Uses arithmetic pseudo-property method recommended by Hugill and van
        Welsenes (1986).

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.
            correlation: Correlation method ("Pitzer" or "Brock-Bird").

        Returns:
            Mixture surface tension in N/m.
        """
        T = T.to("K")

        # Mole fraction for each compound
        Xi = self.Y2X(Yi).magnitude

        # Surface tension for each compound (N/m)
        sti = self.surface_tension(T, correlation=correlation)

        # Mixture surface tension via arithmetic mean, Poling (12-5.2)
        st = mixing_rule(sti, Xi, "arithmetic")

        return st.to("N/m")

    def mixture_thermal_conductivity(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
    ) -> types.Quantity0D:
        """Calculate thermal conductivity of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.

        Returns:
            Thermal conductivity in W/m/K.
        """
        T = T.to("K")
        tc = self.thermal_conductivity(T).to("W/(m*K)").magnitude
        return Units.Quantity(np.sum(Yi.magnitude * tc ** (-2)) ** (-0.5), "W/(m*K)")


__all__ = ["Fuel"]
