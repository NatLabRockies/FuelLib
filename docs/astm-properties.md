# ASTM-Oriented Fuel Properties

FuelLib provides NumPy-based estimates for six properties commonly used when
screening aviation fuels:

| Property | Method | Return value |
| --- | --- | --- |
| Net heat of combustion | `fuel.heat_of_combustion()` | MJ/kg or kJ/mol |
| Liquid specific heat | `fuel.Cl()` | J/kg/K per component |
| Flash point | `fuel.flash_point()` | K for the mixture |
| Freeze point | `fuel.freeze_point()` | K for the mixture |
| Unified Yield Sooting Index | `fuel.ysi()` | Dimensionless mixture YSI |
| Derived Cetane Number | `fuel.dcn()` | Dimensionless mixture DCN |

The YSI and DCN models also provide one-standard-deviation estimates through
`fuel.ysi_uncertainty()` and `fuel.dcn_uncertainty()`.

These methods are predictive correlations. They are useful for screening,
optimization, and comparison, but they do not replace certified ASTM laboratory
measurements.

## Quick start

```python
import numpy as np

import fuellib as fl

fuel = fl.fuel("posf10325")

lhv = fuel.heat_of_combustion()                    # MJ/kg
component_cl = fuel.Cl(298.15)                    # J/kg/K per component
mixture_cl = float(np.sum(fuel.Y_0 * component_cl))
flash_point = fuel.flash_point()                   # K
freeze_point = fuel.freeze_point()                 # K
ysi = fuel.ysi()
dcn = fuel.dcn()
ysi_sigma = fuel.ysi_uncertainty()
dcn_sigma = fuel.dcn_uncertainty()
```

Every mixture method uses `fuel.Y_0` when `Yi` is omitted. A supplied `Yi` must
follow `fuel.compounds`, contain mass fractions, and represent the intended
normalized composition.

## Supporting data

The models use five package-data tables under
`fuellib/data/gcmTableData/`:

- `gcmExtendedTable.csv` supplies carbon and hydrogen atom counts,
  Ruzicka-Domalski liquid heat-capacity coefficients, and Alibakhshi flash-point
  group contributions.
- `property_anchors.csv` supplies selected experimental normal boiling points,
  melting points, and acentric factors. Exact GCxGC-bin matches take precedence;
  a molecular-formula fallback is used only when the bin is not represented.
- `fusion_families.csv` supplies family correlations for entropy of fusion.
- `das_2018_ysi.csv` supplies component YSI values, uncertainty estimates,
  provenance, molecular formulas, and hydrocarbon families.
- `dcn.csv` supplies component DCN values, uncertainty estimates, provenance,
  molecular formulas, and hydrocarbon families.

All files are included in installed distributions by the package-data rules in
`pyproject.toml`.

### Property anchors

Some molecular symmetries and isomer effects are not represented adequately by
the base Constantinou-Gani group decomposition. FuelLib retains its original
`Tb`, `Tm`, and `omega` arrays for existing property models and applies
available experimental anchors only to `Tb_astm`, `Tm_astm`, and `omega_astm`.

If an experimental boiling point is available but an acentric-factor anchor is
not, the implementation attempts a Kesler-Lee closure. The closure is accepted
only for a reduced boiling temperature below 0.90 and an inferred acentric
factor between 0 and 1.2. The source of each final value is available in
`Tb_source`, `Tm_source`, and `omega_source`.

These anchored values affect only the new flash-point and freeze-point models;
they do not alter inputs used by existing FuelLib property correlations.

## Net heat of combustion

### API

```python
fuel.heat_of_combustion(Yi=None, basis="mass")
```

- `Yi`: optional component mass fractions.
- `basis="mass"`: return the mixture lower heating value in MJ/kg.
- `basis="mol"`: return the mole-fraction-weighted value in kJ/mol.

### Model

The method uses a Hess cycle with the Constantinou-Gani ideal-gas formation
enthalpy and a liquid-phase correction. For component $i$,

$$
H_{f,i}^{liq} = H_{f,i}^{gas} - H_{vap,i}.
$$

For a hydrocarbon with $n_C$ carbon atoms and $n_H$ hydrogen atoms,

$$
\Delta H_{c,i} = n_C H_f^{CO_2(g)}
+ \frac{n_H}{2} H_f^{H_2O(g)} - H_{f,i}^{liq}.
$$

The constants used at 298.15 K are

$$
H_f^{CO_2(g)}=-393.51\ \mathrm{kJ/mol}, \qquad
H_f^{H_2O(g)}=-241.83\ \mathrm{kJ/mol}.
$$

Gaseous water gives the net or lower heating value:

$$
LHV_i = -\frac{\Delta H_{c,i}}{MW_i}.
$$

The mass-basis mixture value is linear in mass fraction:

$$
LHV_{mix}=\sum_i Y_i LHV_i.
$$

The molar option converts each component value to kJ/mol and mixes it using
mole fractions.

### Scope and errors

The implementation currently supports hydrocarbons only. It checks the group
decomposition for heteroatom-containing groups and raises
`NotImplementedError` when they contribute. An unsupported `basis` raises the
same exception.

The result is an engineering estimate related to ASTM D4809/D3338 heating-value
characterization; it is not a simulated bomb-calorimeter test.

## Liquid specific heat

### API

```python
fuel.Cl(T, comp_idx=None)
```

- `T`: temperature in K.
- `comp_idx=None`: return one value per component.
- `comp_idx=i`: return the value for component `i`.
- Return units: J/kg/K.

`Cl` is a component property. A mass-weighted ideal mixture estimate can be
formed with:

```python
cl_mix = float(np.sum(Yi * fuel.Cl(T)))
```

### Model

FuelLib uses the Ruzicka-Domalski second-order group-additivity form. The
extended table projects its coefficients onto the Constantinou-Gani group set.
For each component,

$$
C_{p,L,i}(T)=\frac{R}{MW_i}
\left[A_i+B_i\left(\frac{T}{100}\right)
+D_i\left(\frac{T}{100}\right)^2\right].
$$

The component coefficients are group sums:

$$
A_i=\sum_k N_{ik}A_k,\quad
B_i=\sum_k N_{ik}B_k,\quad
D_i=\sum_k N_{ik}D_k.
$$

The correlation is intended for the liquid range between melting and normal
boiling temperatures. Accuracy deteriorates when extrapolated toward the
critical point or outside the calibrated hydrocarbon families.

## Flash point

### API

```python
fuel.flash_point(
    Yi=None,
    method="Alibakhshi",
    mixing="Liaw",
)
```

Supported pure-component methods:

- `"Alibakhshi"` (default)
- `"Alqaheem"`

Supported mixture rules:

- `"Liaw"` (default)
- `"linear"`

The result is in K. Unsupported method or mixing-rule names raise
`NotImplementedError`.

### Pure-component correlations

The default Alibakhshi et al. correlation is

$$
T_{fp,i}=12.14+0.73T_{b,i}+\sum_k N_{ik}\phi_k,
$$

where $T_b$ is the anchored normal boiling point and $\phi_k$ values come from
`gcmExtendedTable.csv`.

The Alqaheem-Riazi alternative is

$$
T_{fp,i}=0.70T_{b,i}.
$$

### Liaw-Chiu mixture rule

Mass fractions are converted to mole fractions. With ideal activity
coefficients, the mixture flash point satisfies

$$
\sum_i x_i
\frac{P_i^{sat}(T_{fp,mix})}{P_i^{sat}(T_{fp,i})}=1.
$$

FuelLib solves this residual with ten Newton iterations initialized from the
mole-fraction-weighted pure-component flash point. Each Newton step is limited
to 20 K.

Saturation pressures use the Lee-Kesler relation:

$$
P_i^{sat}=P_{c,i}\exp\left[f_0(T_r)+\omega_i f_1(T_r)\right],
\qquad T_r=\frac{T}{T_{c,i}},
$$

with

$$
f_0=5.92714-\frac{6.09648}{T_r}-1.28862\ln T_r+0.169347T_r^6,
$$

$$
f_1=15.2518-\frac{15.6875}{T_r}-13.4721\ln T_r+0.43577T_r^6.
$$

The `"linear"` option instead returns

$$
T_{fp,mix}=\sum_i x_iT_{fp,i}.
$$

The ideal Liaw rule does not model nonideal liquid activity coefficients.

## Freeze point

### API

```python
fuel.freeze_point(Yi=None, method="Boehm2022", alpha=1.0)
```

Only `method="Boehm2022"` is supported. The result is in K.

### Fusion properties

For each hydrocarbon family, `fusion_families.csv` defines

$$
\Delta S_{fus,i}=
\max\left[A_f+B_f(n_{C,i}-C_{ref,f}),20\right]
\quad \mathrm{J/(mol\,K)}.
$$

Unclassified compounds use the Walden-rule fallback of 56.5 J/mol/K. Enthalpy
of fusion is then

$$
\Delta H_{fus,i}=T_{m,i}\Delta S_{fus,i}.
$$

The solid-minus-liquid heat-capacity approximation is

$$
\Delta C_{p,i}=-0.35C_{p,L,i}(298.15\ \mathrm{K}),
$$

where liquid heat capacity is converted to a molar basis.

### Solid-liquid equilibrium

For each component $j$, the ideal binary split of that component against the
rest of the mixture gives

$$
\Delta S_{mix,j}=-\frac{R}{x_j}
\left[(1-x_j)\ln(1-x_j)+x_j\ln x_j\right].
$$

FuelLib performs eight fixed-point iterations of the Boehm et al. equation:

$$
T_j =
\frac{\Delta H_{fus,j}+x_j\Delta C_{p,j}(T_{m,j}-T_j)}
{\Delta S_{fus,j}+x_j\Delta C_{p,j}\ln(T_j/T_{m,j})
+\alpha\Delta S_{mix,j}}.
$$

The default `alpha=1.0` applies the full ideal mixing-entropy term. Components
with mole fraction at or below $10^{-6}$ are excluded. The reported mixture
freeze point is the first-crystal temperature:

$$
T_{freeze}=\max_j T_j.
$$

This is an equilibrium screening model. It does not model cooling rate,
supercooling, crystal kinetics, or detailed solid-phase nonideality.

## Unified Yield Sooting Index

### API

```python
fuel.ysi(Yi=None)
fuel.ysi_uncertainty(Yi=None)
```

Both return dimensionless values.

### Lookup and blending

Component values come from the packaged YSI table derived from the
McEnally-Pfefferle Yale YSI Database Volume 2. FuelLib first uses an exact
GCxGC-bin match. For an unknown bin it may use a molecular-formula match, with
n-alkanes preferred when a formula is shared by multiple families.

The mixture value is linear in mole fraction:

$$
YSI_{mix}=\sum_i x_iYSI_i.
$$

If a component remains unmatched, construction fills it using the mean of
available components in the same family. If no family values are available,
the table median is used. `fuel.ysi()` emits a `RuntimeWarning` when a filled
component contributes above a mole fraction of $10^{-6}$.

Per-component provenance and fallback state are available through:

```python
fuel.ysi_source
fuel.ysi_filled
fuel.ysi_err
fuel.ysi_pure
```

Non-measured estimates receive an uncertainty of at least twice their table
error or 15% of their absolute YSI. Family-mean fills receive at least 30% or
10 YSI units.

Assuming independent component errors, uncertainty is propagated as

$$
\sigma_{YSI}=\sqrt{\sum_i(x_i\sigma_i)^2}.
$$

The uncertainty covers component-table errors and fallback inflation. It does
not represent every uncertainty in the linear blending assumption.

## Derived Cetane Number

### API

```python
fuel.dcn(Yi=None, T_ref=288.15)
fuel.dcn_uncertainty(Yi=None, T_ref=288.15)
```

- `T_ref`: reference temperature in K used to calculate component liquid
  densities and volume fractions.
- Return value: DCN on the ASTM D6890 IQT scale.

### Lookup and blending

Component values and uncertainties come from `dcn.csv`. As with YSI, exact
GCxGC-bin matches take precedence and formula fallback is limited to bins not
represented directly in the table.

DCN is mixed by liquid volume fraction. For mass fractions $Y_i$ and component
densities $\rho_i(T_{ref})$,

$$
\phi_i=\frac{Y_i/\rho_i}{\sum_jY_j/\rho_j},
$$

and

$$
DCN_{mix}=\sum_i\phi_iDCN_i.
$$

If a component with positive mass fraction has no DCN assignment, `dcn()`
raises `NotImplementedError` and reports the missing component names. Relevant
component metadata is available through:

```python
fuel.dcn_pure
fuel.dcn_err
fuel.dcn_source
fuel.bin_family
```

Assuming independent component errors, table uncertainty is propagated as

$$
\sigma_{DCN}=\sqrt{\sum_i(\phi_i\sigma_i)^2}.
$$

This does not include uncertainty from the density correlation or the linear
volume-blending assumption.

### C-1 ATJ limitation

The current POSF11498 decomposition represents the dominant isoparaffins using
lightly branched reference compounds. It does not represent the true highly
branched C-1 ATJ isomers closely enough for a defensible C-1 DCN validation.
Consequently, C-1 ATJ DCN is explicitly outside the validation scope of this
transition. The model still returns a numerical estimate for POSF11498, but it
should not be compared with the measured C-1 value until the reference-compound
mapping is addressed independently.

## Composition and fallback behavior

The methods accept optional mass fractions instead of changing the stored fuel
composition:

```python
Yi = fuel.Y_0.copy()
Yi[0] += 0.01
Yi /= Yi.sum()
result = fuel.flash_point(Yi=Yi)
```

The original `fuel.Y_0` remains unchanged. Callers should normalize supplied
compositions and retain the same component ordering as `fuel.compounds`.

Fallback behavior differs by property:

| Property | Missing-data behavior |
| --- | --- |
| Heat of combustion | Rejects non-hydrocarbon group occupancy |
| Liquid specific heat | Uses available group coefficients |
| Flash point | Uses projected group coefficients and anchored `Tb` |
| Freeze point | Uses 56.5 J/mol/K fusion entropy for unclassified families |
| YSI | Fills by family mean or table median and warns when used |
| DCN | Raises if a contributing component has no assignment |

## Examples

### Evaluate an alternate composition

```python
import numpy as np
import fuellib as fl

fuel = fl.fuel("posf10325")
Yi = np.asarray(fuel.Y_0, dtype=float).copy()
Yi *= 0.95
Yi[0] += 0.05
Yi /= Yi.sum()

print(f"LHV:    {fuel.heat_of_combustion(Yi):.3f} MJ/kg")
print(f"Flash:  {fuel.flash_point(Yi):.2f} K")
print(f"Freeze: {fuel.freeze_point(Yi):.2f} K")
print(f"YSI:    {fuel.ysi(Yi):.2f} +/- {fuel.ysi_uncertainty(Yi):.2f}")
print(f"DCN:    {fuel.dcn(Yi):.2f} +/- {fuel.dcn_uncertainty(Yi):.2f}")
```

### Select alternate flash-point options

```python
flash_alibakhshi_liaw = fuel.flash_point(
    method="Alibakhshi",
    mixing="Liaw",
)
flash_alqaheem_linear = fuel.flash_point(
    method="Alqaheem",
    mixing="linear",
)
```

### Inspect component provenance

```python
for name, ysi, source, filled in zip(
    fuel.compounds,
    fuel.ysi_pure,
    fuel.ysi_source,
    fuel.ysi_filled,
):
    print(name, ysi, source, filled)

for name, dcn, source in zip(
    fuel.compounds,
    fuel.dcn_pure,
    fuel.dcn_source,
):
    print(name, dcn, source)
```

## References

- Constantinou, L. and Gani, R. Group-contribution estimation of pure-component
  properties.
- Ruzicka, V. and Domalski, E. S. Estimation of liquid heat capacities by group
  contribution, 1993.
- Alibakhshi et al. Modified group-contribution prediction of pure-component
  flash points, 2015.
- Alqaheem, Y. and Riazi, M. Flash-point correlation for hydrocarbons, 2017.
- Liaw, H.-J. and Chiu, Y.-Y. A general model for predicting mixture flash
  points, 2006.
- Boehm et al. Blend prediction model for the freeze point of jet-fuel-range
  hydrocarbons, 2022.
- Das et al. Unified Yield Sooting Index measurements and correlation,
  *Combustion and Flame* 190, 2018.
- McEnally and Pfefferle, Yale YSI Database Volume 2.
- ASTM D4809 and ASTM D3338 for aviation-fuel heat of combustion.
- ASTM D6890 for ignition quality and Derived Cetane Number by IQT.
