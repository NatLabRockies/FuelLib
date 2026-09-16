"""Fuel class for Group Contribution Method calculations."""

import os
import re

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
from .convert import K2C
from .utility import mixing_rule

# Standard-state gas-phase formation enthalpies at 298.15 K. Using gaseous
# water gives the net (lower) heating value reported for aviation fuels.
_HF_CO2_G_JMOL = -393.51e3
_HF_H2O_G_JMOL = -241.83e3


def _psat_lee_kesler(T, Tc, Pc, omega):
    """Return Lee-Kesler saturation pressures in Pa."""
    Tr = T / Tc
    f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * Tr**6
    f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * Tr**6
    return Pc * np.exp(f0 + omega * f1)


def _fp_alqaheem(Tb):
    """Return Alqaheem-Riazi pure-component flash points in K."""
    return 0.70 * Tb


def _fp_alibakhshi(Tb, phi_sum):
    """Return Alibakhshi pure-component flash points in K."""
    return 12.14 + 0.73 * Tb + phi_sum


def _fp_liaw_ideal_iter(Xi, Tf_i, Tc, Pc, omega, n_iter=10):
    """Solve the ideal Liaw-Chiu mixture flash-point criterion."""
    psat_ref = _psat_lee_kesler(Tf_i, Tc, Pc, omega)

    def residual(T):
        psat_T = _psat_lee_kesler(T, Tc, Pc, omega)
        return np.sum(Xi * psat_T / psat_ref) - 1.0

    T = np.sum(Xi * Tf_i)
    dT = 1.0
    for _ in range(n_iter):
        value = residual(T)
        derivative = (residual(T + dT) - value) / dT
        step = np.clip(value / (derivative + 1e-30), -20.0, 20.0)
        T -= step
    return T


def _boehm2022_iter(x_j, Tm_j, dHfus_j, dSfus_j, dCp_j, alpha=1.0, n_iter=8):
    """Solve Boehm et al. (2022), equation 21, for each component."""
    gas_constant = 8.31446
    x_safe = np.clip(x_j, 1e-6, 1.0 - 1e-6)
    dS_mix = (
        -gas_constant
        / x_safe
        * ((1.0 - x_safe) * np.log(1.0 - x_safe) + x_safe * np.log(x_safe))
    )
    T = Tm_j * np.ones_like(x_safe)
    for _ in range(n_iter):
        T = np.maximum(T, 1.0)
        numerator = dHfus_j + x_safe * dCp_j * (Tm_j - T)
        denominator = dSfus_j + x_safe * dCp_j * np.log(T / Tm_j)
        denominator += alpha * dS_mix
        T = numerator / (denominator + 1e-30)
    return np.where(T > 0, T, -np.inf)


def _freeze_max_over_j(Xi, Tm_i, dHfus_i, dSfus_i, dCp_i, alpha=1.0):
    """Return the first-crystal temperature from the component candidates."""
    candidates = _boehm2022_iter(Xi, Tm_i, dHfus_i, dSfus_i, dCp_i, alpha=alpha)
    return np.max(np.where(Xi > 1e-6, candidates, -np.inf))


def _ysi_mix(Xi, ysi_i):
    """Return a mole-fraction-weighted Unified YSI."""
    return np.sum(Xi * ysi_i)


def _dcn_mix(phi_i, dcn_i):
    """Return a liquid-volume-fraction-weighted DCN."""
    return np.sum(phi_i * dcn_i)


def _cp_liq_rd(T, A, B, D, MW):
    """Return Ruzicka-Domalski liquid heat capacities in J/kg/K."""
    gas_constant = 8.31446
    reduced_temperature = T / 100.0
    cp_molar = gas_constant * (A + B * reduced_temperature + D * reduced_temperature**2)
    return cp_molar / MW


def _lhv_hess(n_C, n_H, Hf, MW):
    """Return component lower heating values in MJ/kg from a Hess cycle."""
    combustion_enthalpy = n_C * _HF_CO2_G_JMOL
    combustion_enthalpy += (n_H / 2.0) * _HF_H2O_G_JMOL - Hf
    return -combustion_enthalpy * 1e-6 / MW


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

    #: Molecular weights in kg/mol. Shape: (num_compounds,)
    MW: np.ndarray

    #: Critical temperatures in K. Shape: (num_compounds,)
    Tc: np.ndarray

    #: Critical pressures in Pa. Shape: (num_compounds,)
    Pc: np.ndarray

    #: Critical volumes in m³/mol. Shape: (num_compounds,)
    Vc: np.ndarray

    #: Boiling temperatures in K. Shape: (num_compounds,)
    Tb: np.ndarray

    #: Melting temperatures in K. Shape: (num_compounds,)
    Tm: np.ndarray

    #: Enthalpy of formation in J/mol. Shape: (num_compounds,)
    Hf: np.ndarray

    #: Gibbs free energy in J/mol. Shape: (num_compounds,)
    Gf: np.ndarray

    #: Enthalpy of vaporization at 298 K in J/mol. Shape: (num_compounds,)
    Hv_stp: np.ndarray

    #: Latent heat of vaporization at 298 K in J/kg. Shape: (num_compounds,)
    Lv_stp: np.ndarray

    #: Molar specific heat at 298 K in J/mol/K. Shape: (num_compounds,)
    Cp_stp: np.ndarray

    #: Molar liquid volume at 298 K in m³/mol. Shape: (num_compounds,)
    Vm_stp: np.ndarray

    #: Acentric factors. Shape: (num_compounds,)
    omega: np.ndarray

    #: Lennard-Jones collision diameters in m. Shape: (num_compounds,)
    sigma: np.ndarray

    #: Lennard-Jones well depths in K. Shape: (num_compounds,)
    epsilonByKB: np.ndarray

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
        df_table = df_table.drop(columns=["Units"])

        self.group_names = [str(column) for column in df_table.columns[1:]]
        self._non_hc_idx = [
            index
            for index, group_name in enumerate(self.group_names)
            if re.search(r"O|N|S|F|I|Cl|Br", group_name)
        ]

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
        self.MW = np.matmul(self.Nij, mwk)  # g/mol
        self.MW *= 1e-3  # Convert to kg/mol

        # T_c (critical temperature)
        self.Tc = 181.128 * np.log(np.matmul(self.Nij, Tck))  # K

        # p_c (critical pressure)
        self.Pc = 1.3705 + (np.matmul(self.Nij, Pck) + 0.10022) ** (-2)  # bar
        self.Pc *= 1e5  # Convert to Pa from bar

        # V_c (critical volume)
        self.Vc = -0.00435 + (np.matmul(self.Nij, Vck))  # m^3/kmol
        self.Vc *= 1e-3  # Convert to m^3/mol

        # T_b (boiling temperature)
        self.Tb = 204.359 * np.log(np.matmul(self.Nij, Tbk))  # K

        # T_m (melting temperature)
        self.Tm = 102.425 * np.log(np.matmul(self.Nij, Tmk))  # K

        # H_f (enthalpy of formation)
        self.Hf = 10.835 + np.matmul(self.Nij, hfk)  # kJ/mol
        self.Hf *= 1e3  # Convert to J/mol

        # G_f (Gibbs free energy)
        self.Gf = -14.828 + np.matmul(self.Nij, gfk)  # kJ/mol
        self.Gf *= 1e3  # Convert to J/mol

        # H_v,stp (enthalpy of vaporization at 298 K)
        self.Hv_stp = 6.829 + (np.matmul(self.Nij, hvk))  # kJ/mol
        self.Hv_stp *= 1e3  # Convert to J/mol

        # omega (accentric factor)
        self.omega = 0.4085 * np.log(np.matmul(self.Nij, wk) + 1.1507) ** (1.0 / 0.5050)

        # V_m (molar liquid volume at 298 K)
        self.Vm_stp = 0.01211 + np.matmul(self.Nij, Vmk)  # m^3/kmol
        self.Vm_stp *= 1e-3  # Convert to m^3/mol

        # C_p,stp (molar specific heat at 298 K)
        self.Cp_stp = np.matmul(self.Nij, cpak) - 19.7779  # J/mol/K

        # Temperature corrections for C_p
        self.Cp_B = np.matmul(self.Nij, cpbk)
        self.Cp_C = np.matmul(self.Nij, cpck)

        # L_v,stp (latent heat of vaporization at 298 K)
        self.Lv_stp = self.Hv_stp / self.MW  # J/kg

        # Additional group properties used by the ASTM correlations.
        self.gcmExtendedFile = os.path.join(gcmtable_dir, "gcmExtendedTable.csv")
        df_ext = pd.read_csv(self.gcmExtendedFile).drop(columns=["Units"])

        def get_ext_row(property_name):
            row = df_ext[df_ext["Property"] == property_name]
            if row.empty:
                raise ValueError(
                    f"Property '{property_name}' not found in extended GCM table."
                )
            return row.iloc[:, 1:].to_numpy().flatten()

        self.n_C = np.matmul(self.Nij, get_ext_row("n_C")).astype(float)
        self.n_H = np.matmul(self.Nij, get_ext_row("n_H")).astype(float)

        # Experimental anchors correct group-contribution boiling and melting
        # points where the molecular symmetry omitted by GCM is important.
        self.Tb_astm = self.Tb.copy()
        self.Tm_astm = self.Tm.copy()
        self.omega_astm = self.omega.copy()
        self.Tb_source = ["gcm"] * self.num_compounds
        self.Tm_source = ["gcm"] * self.num_compounds
        self.omega_source = ["gcm"] * self.num_compounds

        anchor_file = os.path.join(gcmtable_dir, "property_anchors.csv")
        df_anchor = pd.read_csv(anchor_file)

        def anchor_lookup(value_column, source_column):
            by_bin = {}
            by_formula = {}
            family_by_formula = {}
            for _, row in df_anchor.iterrows():
                if pd.isna(row[value_column]):
                    continue
                value = (float(row[value_column]), str(row[source_column]))
                by_bin[row["GCxGC_Bin"]] = value
                formula = row["Formula"]
                previous_family = family_by_formula.get(formula)
                if previous_family is None or (
                    previous_family != "n_alkane" and row["Family"] == "n_alkane"
                ):
                    by_formula[formula] = value
                    family_by_formula[formula] = row["Family"]
            return by_bin, by_formula

        tb_by_bin, tb_by_formula = anchor_lookup("exp_Tb_K", "Tb_source")
        tm_by_bin, tm_by_formula = anchor_lookup("exp_Tm_K", "Tm_source")
        omega_by_bin, omega_by_formula = anchor_lookup("exp_omega", "omega_source")
        known_anchor_bins = set(df_anchor["GCxGC_Bin"])

        def compound_formula(index):
            return f"C{int(self.n_C[index])}H{int(self.n_H[index])}"

        for index, compound in enumerate(self.compounds):
            known_bin = compound in known_anchor_bins
            formula = compound_formula(index)
            tb_hit = tb_by_bin.get(compound) or (
                None if known_bin else tb_by_formula.get(formula)
            )
            tm_hit = tm_by_bin.get(compound) or (
                None if known_bin else tm_by_formula.get(formula)
            )
            omega_hit = omega_by_bin.get(compound) or (
                None if known_bin else omega_by_formula.get(formula)
            )
            if tb_hit is not None:
                self.Tb_astm[index], self.Tb_source[index] = tb_hit
            if tm_hit is not None:
                self.Tm_astm[index], self.Tm_source[index] = tm_hit
            if omega_hit is not None:
                self.omega_astm[index], self.omega_source[index] = omega_hit

        needs_omega_closure = np.array(
            [
                tb_source != "gcm" and omega_source == "gcm"
                for tb_source, omega_source in zip(self.Tb_source, self.omega_source)
            ]
        )
        if np.any(needs_omega_closure):
            reduced_boiling_temperature = self.Tb_astm / self.Tc
            f0 = (
                5.92714
                - 6.09648 / reduced_boiling_temperature
                - 1.28862 * np.log(reduced_boiling_temperature)
                + 0.169347 * reduced_boiling_temperature**6
            )
            f1 = (
                15.2518
                - 15.6875 / reduced_boiling_temperature
                - 13.4721 * np.log(reduced_boiling_temperature)
                + 0.43577 * reduced_boiling_temperature**6
            )
            omega_closure = (-np.log(self.Pc / 101325.0) - f0) / f1
            valid_closure = (
                (reduced_boiling_temperature < 0.90)
                & (omega_closure > 0.0)
                & (omega_closure < 1.2)
            )
            apply_closure = needs_omega_closure & valid_closure
            self.omega_astm = np.where(apply_closure, omega_closure, self.omega_astm)
            for index in np.where(apply_closure)[0]:
                self.omega_source[index] = "kesler_lee_closure"

        self.Cp_L_A = np.matmul(self.Nij, get_ext_row("rd_A")).astype(float)
        self.Cp_L_B = np.matmul(self.Nij, get_ext_row("rd_B")).astype(float)
        self.Cp_L_D = np.matmul(self.Nij, get_ext_row("rd_D")).astype(float)
        self.alibakhshi_phi = np.matmul(self.Nij, get_ext_row("alibakhshi_phi")).astype(
            float
        )

        fusion_file = os.path.join(gcmtable_dir, "fusion_families.csv")
        fusion_table = pd.read_csv(fusion_file)
        fusion_families = {
            row["Family"]: (
                float(row["dSfus_A"]),
                float(row["dSfus_B"]),
                float(row["C_ref"]),
            )
            for _, row in fusion_table.iterrows()
        }
        cp_liquid_298 = (
            _cp_liq_rd(298.15, self.Cp_L_A, self.Cp_L_B, self.Cp_L_D, self.MW) * self.MW
        )
        self.dCp = -0.35 * cp_liquid_298

        def priority_formula_map(table, columns):
            values = {}
            family_by_formula = {}
            for _, row in table.iterrows():
                formula = row["Formula"]
                previous_family = family_by_formula.get(formula)
                if previous_family is None or (
                    previous_family != "n_alkane" and row["Family"] == "n_alkane"
                ):
                    values[formula] = tuple(row[column] for column in columns)
                    family_by_formula[formula] = row["Family"]
            return values

        self.dasYsiFile = os.path.join(gcmtable_dir, "das_2018_ysi.csv")
        ysi_table = pd.read_csv(self.dasYsiFile)
        ysi_by_bin = dict(zip(ysi_table["GCxGC_Bin"], ysi_table["YSI"]))
        ysi_source_by_bin = dict(zip(ysi_table["GCxGC_Bin"], ysi_table["Source"]))
        ysi_error_by_bin = dict(zip(ysi_table["GCxGC_Bin"], ysi_table["YSI_err"]))
        ysi_family_by_bin = dict(zip(ysi_table["GCxGC_Bin"], ysi_table["Family"]))
        ysi_by_formula = priority_formula_map(ysi_table, ["YSI", "Source", "YSI_err"])
        known_ysi_bins = set(ysi_table["GCxGC_Bin"])

        self.ysi_pure = np.full(self.num_compounds, np.nan, dtype=float)
        self.ysi_err = np.full(self.num_compounds, np.nan, dtype=float)
        self.ysi_source = ["unknown"] * self.num_compounds
        for index, compound in enumerate(self.compounds):
            known_bin = compound in known_ysi_bins
            if known_bin and not np.isnan(ysi_by_bin[compound]):
                self.ysi_pure[index] = ysi_by_bin[compound]
                self.ysi_err[index] = ysi_error_by_bin[compound]
                self.ysi_source[index] = ysi_source_by_bin[compound]
            elif not known_bin:
                hit = ysi_by_formula.get(compound_formula(index))
                if hit is not None and not np.isnan(hit[0]):
                    self.ysi_pure[index] = hit[0]
                    self.ysi_source[index] = f"formula_fallback:{hit[1]}"
                    self.ysi_err[index] = hit[2]
            if not np.isnan(self.ysi_pure[index]) and not str(
                self.ysi_source[index]
            ).startswith(("measured", "formula_fallback:measured")):
                self.ysi_err[index] = max(
                    2.0 * float(np.nan_to_num(self.ysi_err[index])),
                    0.15 * abs(self.ysi_pure[index]),
                )

        self.ysi_filled = np.zeros(self.num_compounds, dtype=bool)
        for index in np.where(np.isnan(self.ysi_pure))[0]:
            family = ysi_family_by_bin.get(self.compounds[index])
            family_values = [
                self.ysi_pure[other]
                for other in range(self.num_compounds)
                if not np.isnan(self.ysi_pure[other])
                and ysi_family_by_bin.get(self.compounds[other]) == family
            ]
            if family_values:
                fill = float(np.mean(family_values))
            else:
                fill = float(np.nanmedian(ysi_table["YSI"]))
            self.ysi_pure[index] = fill
            self.ysi_err[index] = max(0.30 * abs(fill), 10.0)
            self.ysi_source[index] = "family_mean_fill"
            self.ysi_filled[index] = True

        self.dcnFile = os.path.join(gcmtable_dir, "dcn.csv")
        dcn_table = pd.read_csv(self.dcnFile)
        dcn_by_bin = dict(zip(dcn_table["GCxGC_Bin"], dcn_table["DCN"]))
        dcn_source_by_bin = dict(zip(dcn_table["GCxGC_Bin"], dcn_table["Source"]))
        dcn_error_by_bin = dict(zip(dcn_table["GCxGC_Bin"], dcn_table["DCN_err"]))
        family_by_bin = dict(zip(dcn_table["GCxGC_Bin"], dcn_table["Family"]))
        dcn_by_formula = priority_formula_map(
            dcn_table, ["DCN", "Source", "DCN_err", "Family"]
        )
        known_dcn_bins = set(dcn_table["GCxGC_Bin"])

        self.dcn_pure = np.full(self.num_compounds, np.nan, dtype=float)
        self.dcn_err = np.full(self.num_compounds, np.nan, dtype=float)
        self.dcn_source = ["unknown"] * self.num_compounds
        self.bin_family = ["unknown"] * self.num_compounds
        for index, compound in enumerate(self.compounds):
            known_bin = compound in known_dcn_bins
            if known_bin and not np.isnan(dcn_by_bin[compound]):
                self.dcn_pure[index] = dcn_by_bin[compound]
                self.dcn_err[index] = dcn_error_by_bin[compound]
                self.dcn_source[index] = dcn_source_by_bin[compound]
                self.bin_family[index] = family_by_bin[compound]
                continue
            if known_bin:
                continue
            hit = dcn_by_formula.get(compound_formula(index))
            if hit is not None and not np.isnan(hit[0]):
                self.dcn_pure[index] = hit[0]
                self.dcn_source[index] = f"formula_fallback:{hit[1]}"
                self.dcn_err[index] = hit[2]
                self.bin_family[index] = hit[3]

        self.dSfus = np.full(self.num_compounds, 56.5)
        for index, family in enumerate(self.bin_family):
            if family in fusion_families:
                coefficient, slope, reference_carbon = fusion_families[family]
                self.dSfus[index] = max(
                    coefficient + slope * (self.n_C[index] - reference_carbon),
                    20.0,
                )
        self.dHfus = self.dSfus * self.Tm_astm

        # Lennard-Jones parameters for diffusion calculations (Tee et al. 1966)
        self.epsilonByKB = (0.7915 + 0.1693 * self.omega) * self.Tc  # K
        Pc_atm = self.Pc / 101325  # atm
        self.sigma = (2.3551 - 0.0874 * self.omega) * (self.Tc / Pc_atm) ** (
            1.0 / 3
        )  # Angstroms
        self.sigma *= 1e-10  # Convert from Angstroms to m

    # -------------------------------------------------------------------------
    # Member functions
    # -------------------------------------------------------------------------
    def mean_molecular_weight(self, Yi):
        """
        Calculate the mean molecular weight of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :return: Mean molecular weight of the mixture in kg/mol.
        :rtype: float
        """
        if np.sum(Yi) != 0:
            Mbar = 1 / np.sum(Yi / self.MW)  # mean molar weight of the mixture
        else:
            Mbar = 0.0

        return Mbar

    def mass2Y(self, mass):
        """
        Calculate the mass fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Normalize to get group mole fractions
        total_mass = np.sum(mass)
        if total_mass != 0:
            Yi = mass / total_mass
        else:
            Yi = np.zeros_like(self.MW)

        return Yi

    def mass2X(self, mass):
        """
        Calculate the mole fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Calculate the number of moles for each compound
        num_mole = mass / self.MW

        # Normalize to get group mole fractions
        total_moles = np.sum(num_mole)
        if total_moles != 0:
            Xi = num_mole / total_moles
        else:
            Xi = np.zeros_like(self.MW)

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
            Yi = mass / total_mass
        else:
            Yi = np.zeros_like(self.MW)

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
            Xi = Mbar * Yi / self.MW
        else:
            Xi = np.zeros_like(self.MW)

        return Xi

    def density(self, T, comp_idx=None):
        """
        Calculate the density of each component at temperature T.

        :param T: Temperature of the mixture in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Density of each compound in kg/m^3.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            MW = self.MW  # kg/mol
            Vm = self.molar_liquid_vol(T)  # m^3/mol
        else:
            MW = self.MW[comp_idx]  # kg/mol
            Vm = self.molar_liquid_vol(T, comp_idx=comp_idx)  # m^3/mol

        rho = MW / Vm  # kg/m^3
        return rho

    def viscosity_kinematic(self, T, comp_idx=None):
        """
        Calculate the viscosity using Dutt's equation.

        :meta private: This uses Dutt's equation (4.23) from "Viscosity of Liquids".
        :meta private: The equation predicts viscosity in mm^2/s and is converted to SI units.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Viscosity of each component in m^2/s.
        :rtype: np.ndarray
        """

        # Convert temperature to Celsius
        T_cels = K2C(T)
        if comp_idx is None:
            Tb_cels = K2C(self.Tb)
        else:
            Tb_cels = K2C(self.Tb[comp_idx])

        # RHS of Dutt's equation (4.23) in Viscosity of Liquids
        rhs = -3.0171 + (442.78 + 1.6452 * Tb_cels) / (T_cels + 239 - 0.19 * Tb_cels)
        nu_i = np.exp(rhs)  # Viscosity in mm^2/s

        # Convert to SI (m^2/s)
        nu_i = nu_i * 1e-6

        return nu_i

    def viscosity_dynamic(self, T, comp_idx=None):
        """
        Calculate liquid dynamic viscosity based on droplet temperature and density.

        :meta private: Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Dynamic viscosity in Pa*s.
        :rtype: np.ndarray
        """

        nu_i = self.viscosity_kinematic(T, comp_idx=comp_idx)  # m^2/s
        rho_i = self.density(T, comp_idx=comp_idx)  # kg/m^3
        mu_i = nu_i * rho_i  # Pa*s
        return mu_i

    def Cp(self, T, comp_idx=None):
        """
        Compute molar specific heat capacity at a given temperature.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar specific heat capacity in J/mol/K.
        :rtype: np.ndarray
        """

        theta = (T - 298) / 700
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
        Compute liquid specific heat capacity at a given temperature.

        Uses the Ruzicka-Domalski second-order group-additivity correlation
        for the liquid phase. The coefficients are projected onto the
        Constantinou-Gani group set in ``gcmExtendedTable.csv``. The model is
        calibrated from the melting temperature to the normal boiling
        temperature; extrapolation deteriorates near the critical point.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Liquid specific heat capacity in J/kg/K.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            A = self.Cp_L_A
            B = self.Cp_L_B
            D = self.Cp_L_D
            MW = self.MW
        else:
            A = self.Cp_L_A[comp_idx]
            B = self.Cp_L_B[comp_idx]
            D = self.Cp_L_D[comp_idx]
            MW = self.MW[comp_idx]
        return _cp_liq_rd(T, A, B, D, MW)

    def psat(self, T, comp_idx=None, correlation="Lee-Kesler"):
        """
        Compute saturated vapor pressure.

        :meta private: Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Saturated vapor pressure in Pa.
        :rtype: np.ndarray
        """

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

        psat = Pc * rhs
        return psat

    def psat_antoine_coeffs(self, Tvals=None, units="mks", correlation="Lee-Kesler"):
        """
        Estimate Antoine coefficients for vapor pressure of an individual compound.

        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, Tb_i]).
        :type Tvals: np.ndarray, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: 4 np.ndarrays
        """

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, Tb_i] for each compound.")
            # Initialize as zeros for now, calculated for each compound later
            T = np.zeros(20)
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        elif len(Tvals) > 2:
            T = Tvals
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
                T = np.linspace(273.15, self.Tb[i], 20)
            Pvals = np.zeros_like(T)
            for k in range(len(T)):
                Pvals[k] = 1 / D * self.psat(T[k], correlation=correlation)[i]

            logP = np.log10(Pvals)
            popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])
            A[i], B[i], C[i] = popt
        D = D + np.zeros(self.num_compounds)  # make D an array
        return A, B, C, D

    def molar_liquid_vol(self, T, comp_idx=None):
        """
        Compute molar liquid volume with temperature correction.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar liquid volume in m^3/mol.
        :rtype: np.ndarray
        """

        Tstp = 298.0
        if comp_idx is None:
            Tc = self.Tc
            omega = self.omega
            Vm_stp = self.Vm_stp
        else:
            Tc = np.array([self.Tc[comp_idx]])
            omega = np.array([self.omega[comp_idx]])
            Vm_stp = np.array([self.Vm_stp[comp_idx]])
        phi = np.zeros_like(Tc)
        for i in range(len(Tc)):
            if T > Tc[i]:
                phi[i] = -((1 - (Tstp / Tc[i])) ** (2.0 / 7.0))
            else:
                phi[i] = ((1 - (T / Tc[i])) ** (2.0 / 7.0)) - (
                    (1 - (Tstp / Tc[i])) ** (2.0 / 7.0)
                )
        z = 0.29056 - 0.08775 * omega
        Vmi = Vm_stp * np.power(z, phi)
        if comp_idx is not None:
            Vmi = Vmi[0]
        return Vmi

    def latent_heat_vaporization(self, T, comp_idx=None):
        """
        Calculate latent heat of vaporization adjusted for temperature.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Latent heat of vaporization in J/kg.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            Tc = self.Tc
            Tb = self.Tb
            Lv_stp = self.Lv_stp
        else:
            Tc = np.array([self.Tc[comp_idx]])
            Tb = np.array([self.Tb[comp_idx]])
            Lv_stp = np.array([self.Lv_stp[comp_idx]])

        # Reduced temperatures
        Tr = T / Tc
        Trb = Tb / Tc

        Lvi = np.zeros_like(Tc)
        for i in range(len(Tc)):
            if T > Tc[i]:
                Lvi[i] = 0.0
            else:
                Lvi[i] = Lv_stp[i] * (((1.0 - Tr[i]) / (1.0 - Trb[i])) ** 0.38)

        if comp_idx is not None:
            Lvi = Lvi[0]
        return Lvi

    def heat_of_combustion(self, Yi=None, basis="mass"):
        """
        Compute the net heat of combustion (lower heating value) of the fuel.

        Uses a Hess cycle on the Constantinou-Gani ideal-gas enthalpy of
        formation with hydrocarbon combustion stoichiometry. The reactant
        enthalpy is shifted to the liquid phase using the enthalpy of
        vaporization, and gaseous water gives the lower heating value reported
        for aviation fuels by ASTM D4809 and D3338. Mixture values are linear
        in mass fraction.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :param basis: ``"mass"`` returns MJ/kg; ``"mol"`` returns kJ/mol.
        :type basis: str, optional
        :return: Net heat of combustion of the mixture.
        :rtype: float
        :raises NotImplementedError: If the fuel is not hydrocarbon-only or
            if ``basis`` is unknown.
        """
        if Yi is None:
            Yi = self.Y_0
        Yi = np.asarray(Yi, dtype=float)

        non_hydrocarbon_groups = [
            index for index in self._non_hc_idx if index < self.Nij.shape[1]
        ]
        if np.any(self.Nij[:, non_hydrocarbon_groups] != 0):
            raise NotImplementedError(
                "heat_of_combustion currently supports hydrocarbon-only fuels. "
                "Detected non-zero heteroatom group occupancy in the "
                f"decomposition of '{self.name}'."
            )

        formation_enthalpy_liquid = self.Hf - self.Hv_stp
        component_lhv = _lhv_hess(
            self.n_C, self.n_H, formation_enthalpy_liquid, self.MW
        )
        if basis.casefold() == "mass":
            return float(np.sum(Yi * component_lhv))
        if basis.casefold() == "mol":
            Xi = self.Y2X(Yi)
            component_lhv_kjmol = component_lhv * self.MW * 1e3
            return float(np.sum(Xi * component_lhv_kjmol))
        raise NotImplementedError(
            f"heat_of_combustion basis '{basis}' not supported "
            "(use 'mass' for MJ/kg or 'mol' for kJ/mol)."
        )

    def freeze_point(self, Yi=None, method="Boehm2022", alpha=1.0):
        """
        Compute the freeze point of the fuel using solid-liquid equilibrium.

        The Boehm et al. (2022) equation 21 model is solved for every
        component, and the highest candidate temperature marks the first
        crystal on cooling. Per-component fusion entropies come from the
        family correlations in ``fusion_families.csv``; unclassified
        compounds use the Walden-rule value of 56.5 J/mol/K. The default
        ``alpha=1`` gives the classical ideal-solution entropy term.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :param method: Freeze-point model. Only ``"Boehm2022"`` is supported.
        :type method: str, optional
        :param alpha: Scaling applied to the ideal mixing-entropy term.
        :type alpha: float, optional
        :return: Mixture freeze point in K.
        :rtype: float
        :raises NotImplementedError: If ``method`` is not ``"Boehm2022"``.
        """
        if Yi is None:
            Yi = self.Y_0
        if method.casefold() != "boehm2022":
            raise NotImplementedError(
                f"freeze_point method '{method}' not supported (use 'Boehm2022')."
            )
        Xi = self.Y2X(np.asarray(Yi, dtype=float))
        return float(
            _freeze_max_over_j(
                Xi,
                self.Tm_astm,
                self.dHfus,
                self.dSfus,
                self.dCp,
                alpha=alpha,
            )
        )

    def flash_point(self, Yi=None, method="Alibakhshi", mixing="Liaw"):
        """
        Compute the flash point of the fuel mixture.

        Pure-component values use either the Alibakhshi et al. (2015) group
        contribution model or the Alqaheem-Riazi correlation. Mixtures use
        the ideal Liaw-Chiu modified Le Chatelier criterion by default, with
        a mole-fraction-linear rule available as a simpler alternative.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :param method: Pure-component model, ``"Alibakhshi"`` or
            ``"Alqaheem"``.
        :type method: str, optional
        :param mixing: Mixture rule, ``"Liaw"`` or ``"linear"``.
        :type mixing: str, optional
        :return: Mixture flash point in K.
        :rtype: float
        :raises NotImplementedError: If ``method`` or ``mixing`` is unknown.
        """
        if Yi is None:
            Yi = self.Y_0
        if method.casefold() == "alibakhshi":
            component_flash_points = _fp_alibakhshi(self.Tb_astm, self.alibakhshi_phi)
        elif method.casefold() == "alqaheem":
            component_flash_points = _fp_alqaheem(self.Tb_astm)
        else:
            raise NotImplementedError(
                f"flash_point method '{method}' not supported "
                "(use 'Alibakhshi' or 'Alqaheem')."
            )

        Xi = self.Y2X(np.asarray(Yi, dtype=float))
        if mixing.casefold() == "linear":
            return float(np.sum(Xi * component_flash_points))
        if mixing.casefold() == "liaw":
            return float(
                _fp_liaw_ideal_iter(
                    Xi,
                    component_flash_points,
                    self.Tc,
                    self.Pc,
                    self.omega_astm,
                )
            )
        raise NotImplementedError(
            f"flash_point mixing rule '{mixing}' not supported "
            "(use 'Liaw' or 'linear')."
        )

    def ysi(self, Yi=None):
        """
        Compute the Unified Yield Sooting Index of the mixture.

        Component values come from the McEnally-Pfefferle Yale YSI Database
        Volume 2 and are mixed linearly by mole fraction. Missing tabulated
        values are filled during construction from the fuel-level family mean;
        this method warns when such a value contributes to the result.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :return: Mole-fraction-weighted mixture Unified YSI.
        :rtype: float
        """
        if Yi is None:
            Yi = self.Y_0
        Xi = self.Y2X(np.asarray(Yi, dtype=float))
        contributing = self.ysi_filled & (Xi > 1e-6)
        if np.any(contributing):
            import warnings

            names = [self.compounds[index] for index in np.where(contributing)[0]]
            warnings.warn(
                f"YSI for '{self.name}' uses family-mean fills for {names} "
                "(no tabulated value); see ysi_err and ysi_uncertainty().",
                RuntimeWarning,
                stacklevel=2,
            )
        return float(_ysi_mix(Xi, self.ysi_pure))

    def ysi_uncertainty(self, Yi=None):
        """
        Compute one-sigma uncertainty of the mixture YSI.

        Independent component errors are propagated through the linear
        mole-fraction blending rule. Extrapolated and filled component values
        carry inflated uncertainties recorded in ``self.ysi_err``.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :return: One-sigma uncertainty of the mixture Unified YSI.
        :rtype: float
        """
        if Yi is None:
            Yi = self.Y_0
        Xi = self.Y2X(np.asarray(Yi, dtype=float))
        error = np.nan_to_num(self.ysi_err, nan=0.3 * np.nanmean(self.ysi_pure))
        return float(np.sqrt(np.sum((Xi * error) ** 2)))

    def dcn_uncertainty(self, Yi=None, T_ref=288.15):
        """
        Compute one-sigma uncertainty of the mixture Derived Cetane Number.

        Independent component errors are propagated through the linear liquid
        volume-fraction blending rule. This represents table uncertainty only;
        uncertainty in the blending rule itself is not included.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :param T_ref: Reference temperature in K for volume fractions.
        :type T_ref: float, optional
        :return: One-sigma uncertainty of the mixture DCN.
        :rtype: float
        """
        if Yi is None:
            Yi = self.Y_0
        Yi = np.asarray(Yi, dtype=float)
        component_volume = Yi / self.density(T_ref)
        total_volume = np.sum(component_volume)
        if total_volume > 0:
            volume_fraction = component_volume / total_volume
        else:
            volume_fraction = np.zeros_like(component_volume)
        error = np.nan_to_num(self.dcn_err, nan=8.0)
        return float(np.sqrt(np.sum((volume_fraction * error) ** 2)))

    def dcn(self, Yi=None, T_ref=288.15):
        """
        Compute the Derived Cetane Number of the fuel mixture.

        Component values in ``dcn.csv`` use the ASTM D6890 IQT scale and are
        mixed linearly by liquid volume fraction. Volume fractions are derived
        from mass fractions using component liquid densities at ``T_ref``.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :param T_ref: Reference temperature in K for liquid densities.
        :type T_ref: float, optional
        :return: Volume-fraction-weighted mixture DCN.
        :rtype: float
        :raises NotImplementedError: If a contributing compound has no DCN.
        """
        if Yi is None:
            Yi = self.Y_0
        Yi = np.asarray(Yi, dtype=float)
        missing = np.isnan(self.dcn_pure)
        contributing = missing & (Yi > 0.0)
        if np.any(contributing):
            names = [self.compounds[index] for index in np.where(contributing)[0]]
            raise NotImplementedError(
                "DCN is not tabulated for the following compounds in "
                f"'{self.name}' (mass fraction > 0): {names}."
            )

        component_volume = Yi / self.density(T_ref)
        total_volume = np.sum(component_volume)
        if total_volume > 0:
            volume_fraction = component_volume / total_volume
        else:
            volume_fraction = np.zeros_like(component_volume)
        component_dcn = np.where(missing, 0.0, self.dcn_pure)
        return float(_dcn_mix(volume_fraction, component_dcn))

    def diffusion_coeff(
        self,
        p,
        T,
        sigma_gas=3.62e-10,
        epsilonByKB_gas=97.0,
        MW_gas=28.97e-3,
        correlation="Tee",
    ):
        """
        Compute diffusion coefficients using Lennard-Jones parameters.

        :meta private: Uses Wilke and Lee method (Poling, equation 11-4.1).
        :meta private: Ambient gas defaults to air parameters.

        :param p: Pressure in Pa.
        :type p: float
        :param T: Temperature in Kelvin.
        :type T: float
        :param sigma_gas: Collision diameter in m.
        :type sigma_gas: float, optional
        :param epsilonByKB_gas: Well depth over Boltzmann constant, in K.
        :type epsilonByKB_gas: float, optional
        :param MW_gas: Mean molecular weight of ambient gas in kg/mol.
        :type MW_gas: float, optional
        :param correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
        :type correlation: str, optional
        :return: Diffusion coefficient.
        :rtype: np.ndarray
        """

        # Method of Tee for calculating liquid sigma and epsilon
        if correlation.casefold() == "Tee".casefold():
            sigma_i = self.sigma * 1e10  # convert from m to Angstroms
            epsilonByKB_i = self.epsilonByKB  # K
        else:
            # Method of Wilke & Lee calculating liquid sigma and epsilon
            Vmb_i = np.zeros_like(self.Tb)
            for n in range(self.num_compounds):
                Vmb_i[n] = self.molar_liquid_vol(self.Tb[n])[n] * 1e6  # cm^3/mol
            sigma_i = 1.18 * Vmb_i ** (1 / 3)  # Angstroms, Poling (11-4.2)
            epsilonByKB_i = 1.15 * self.Tb  # K , Poling (11-4.3)

        # Compute binary sigma and epsilon
        sigma_gas = sigma_gas * 1e10  # convert from m to Angstroms
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
        MW_gas = MW_gas * 1e3
        MW_i = self.MW * 1e3
        M_AB_i = 2 * (MW_i * MW_gas) / (MW_i + MW_gas)  # g/mol, see Poling (11-3.1)

        # Convert pressure from Pa to bar
        p = p * 1e-5  # bar

        # Binary diffusion coefficients, Poling (11-4.1)
        D_AB_i = (
            1e-3
            * (3.03 - 0.98 / (M_AB_i**0.5))
            * (T**1.5)
            / (p * M_AB_i**0.5 * sigmaAB_i**2 * omegaD_i)
        )  # cm^2/s
        D_AB_i = D_AB_i * 1e-4  # Convert to m^2/s

        return D_AB_i

    def surface_tension(self, T, comp_idx=None, correlation="Brock-Bird"):
        """
        Calculate surface tension of each compound at a given temperature.

        :meta private: Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Brock-Bird" or "Pitzer").
        :type correlation: str, optional
        :return: Surface tension in N/m.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            Tc = self.Tc
            Pc = self.Pc
            Tb = self.Tb
            omega = self.omega
        else:
            Tc = np.array([self.Tc[comp_idx]])
            Pc = np.array([self.Pc[comp_idx]])
            Tb = np.array([self.Tb[comp_idx]])
            omega = np.array([self.omega[comp_idx]])
        Tr = T / Tc
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

        st = st * 1e-3  # Convert from dyn/cm to N/m
        if comp_idx is not None:
            st = st[0]

        return st

    def thermal_conductivity(self, T, comp_idx=None):
        """
        Calculate thermal conductivity at a given temperature.

        :meta private: Uses Latini et al. method (Poling equation 10-9.1).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Thermal conductivity in W/m/K.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            MW = self.MW
            Tc = self.Tc
            Tb = self.Tb
            fam = self.fam
        else:
            MW = np.array([self.MW[comp_idx]])
            Tc = np.array([self.Tc[comp_idx]])
            Tb = np.array([self.Tb[comp_idx]])
            fam = np.array([self.fam[comp_idx]])

        Astar = 0.00350 + np.zeros_like(Tc)
        alpha = 1.2
        beta = 0.5 + np.zeros_like(Tc)
        gamma = 0.167
        MW_beta = MW * 1e3  # convert from kg/mol to g/mol
        Tr = T / Tc

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
        return tc

    # --- Mixture functions ---
    def mixture_density(self, Yi, T):
        """
        Calculate mixture density at a given temperature.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :return: Mixture density in kg/m^3.
        :rtype: float
        """
        MW = self.MW  # Molecular weights of each component (kg/mol)
        Vmi = self.molar_liquid_vol(T)  # Molar volume of each component (m^3/mol)

        # Calculate density (kg/m^3)
        rho = Yi @ (MW / Vmi)

        return rho

    def mixture_kinematic_viscosity(self, Yi, T, correlation="Kendall-Monroe"):
        """
        Calculate kinematic viscosity of the mixture.

        :meta private: Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture kinematic viscosity in m^2/s.
        :rtype: float
        """
        nu_i = self.viscosity_kinematic(T)  # Viscosities of individual components

        # Calculate mole fractions for each species
        Xi = self.Y2X(Yi)

        if correlation.casefold() == "Arrhenius".casefold():
            # Arrhenius mixing correlation
            nu = np.exp(np.sum(Xi * np.log(nu_i)))
        else:
            # Default: Kendall-Monroe mixing correlation
            nu = np.sum(Xi * (nu_i ** (1.0 / 3.0))) ** (3.0)

        return nu

    def mixture_dynamic_viscosity(self, Yi, T, correlation="Kendall-Monroe"):
        """
        Calculate dynamic viscosity of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture dynamic viscosity in Pa*s.
        :rtype: float
        """

        nu = self.mixture_kinematic_viscosity(Yi, T, correlation=correlation)
        rho = self.mixture_density(Yi, T)

        return rho * nu

    def mixture_vapor_pressure(self, Yi, T, correlation="Lee-Kesler"):
        """
        Calculate vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Mixture vapor pressure in Pa.
        :rtype: float
        """

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Saturated vapor pressure for each compound (Pa)
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
        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, min(Tb)]).
        :type Tvals: np.ndarray, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: float
        """

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, min(Tb_mix)] for mixture.")
            # Initialize as zeros for now, calculated for each compound later
            X = self.Y2X(Yi)
            Tb = mixing_rule(self.Tb, X)
            T = np.linspace(273.15, np.min(Tb), 20)
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        elif len(Tvals) > 2:
            T = Tvals
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

        Pvals = np.zeros_like(T)
        for k in range(len(T)):
            Pvals[k] = (
                self.mixture_vapor_pressure(Yi, T[k], correlation=correlation) / D
            )

        logP = np.log10(Pvals)
        popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])  # initial guess
        A, B, C = popt

        return A, B, C, D

    def mixture_surface_tension(self, Yi, T, correlation="Brock-Bird"):
        """
        Calculate surface tension of the mixture.

        :meta private: Uses arithmetic pseudo-property method recommended by Hugill and van Welsenes (1986).

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Correlation method ("Pitzer" or "Brock-Bird").
        :type correlation: str, optional
        :return: Mixture surface tension in N/m.
        :rtype: float
        """

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Surface tension for each compound (N/m)
        sti = self.surface_tension(T, correlation=correlation)

        # Mixture surface tension via arithmetic mean, Poling (12-5.2)
        st = mixing_rule(sti, Xi, "arithmetic")

        return st

    def mixture_thermal_conductivity(self, Yi, T):
        """
        Calculate thermal conductivity of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :return: Thermal conductivity in W/m/K.
        :rtype: float
        """
        tc = self.thermal_conductivity(T)
        return np.sum(Yi * tc ** (-2)) ** (-0.5)


__all__ = ["fuel"]
