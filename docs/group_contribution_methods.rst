Group Contribution Methods
==========================

Group contribution methods (GCMs) estimate the properties of a pure compound
from the structural groups that make up its molecule. Each group contributes a
predefined value to a target property, and the property is a function of the
sum of these contributions. GCMs need only the structure of a compound, so
they can predict properties of compounds that have not been measured. FuelLib
uses GCMs for the compound properties that are not available as
:ref:`reference data <sec-reference-properties>` (see
:ref:`sec-prediction-methodology`).

.. _sec-gcm-core-concepts:

Core concepts
-------------

The groups of a GCM are defined by the method. *First-order* groups describe
each atom with its attached hydrogens and bonds (e.g., ``CH3``, ``CH2``,
``ACH``), and every non-hydrogen atom of a compound belongs to exactly one
first-order group. *Second-order* groups describe larger structures, such as
rings or adjacent branches, and correct the first-order estimate. For a
mixture of :math:`N_c` compounds, the group decomposition is a matrix whose
entry in row *i* and column *k* is the number of groups *k* in compound *i*.

FuelLib includes four GCMs, each in its own module of :mod:`fuellib.gcm`:

.. _tab-gcm-methods:

.. table:: GCMs in FuelLib.
   :widths: auto
   :align: center

   ==============  ==============================  ================================================  ============================
   Method          Module                          Properties                                        Decomposition
   ==============  ==============================  ================================================  ============================
   ``gani``        :mod:`fuellib.gcm.gani`         Critical properties, boiling and melting points,  ``ref_gani.csv``
                                                   enthalpies, acentric factor, molar volume, and
                                                   heat capacity
   ``nannoolal``   :mod:`fuellib.gcm.nannoolal`    Liquid viscosity parameters                       ``ref_nannoolal.csv``
   ``unifac``      :mod:`fuellib.gcm.unifac`       Volume and surface parameters, activity           ``ref_unifac.csv``
                                                   coefficients
   ``boehm``       :mod:`fuellib.gcm.boehm`        Entropy of fusion                                 ``Family`` and carbon number
   ==============  ==============================  ================================================  ============================

The group contributions of each method are stored in ``fuellib/gcm/<method>.csv``,
and the group decompositions of the :doc:`reference_compounds` are stored in
``fuellib/data/ref_<method>.csv``. Each method is registered in a
:class:`~fuellib.gcm.core.GCMRegistry` that evaluates its properties for a
:class:`~fuellib.fuel.core.Fuel` (see :ref:`sec-gcm-registry`):

.. code-block:: python

   from fuellib import Fuel
   from fuellib.gcm import GCMRegistry

   fuel = Fuel("posf10325")
   gani = GCMRegistry.get_gcm("gani")
   gani.predict("tc", fuel)   # Quantity1D of critical temperatures

Decomposing SMILES strings
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``gani``, ``nannoolal``, and ``unifac`` modules each provide a
``decompose`` function that builds a group decomposition from a SMILES string
with RDKit SMARTS patterns. It returns a single-row :class:`pandas.DataFrame`,
indexed by the SMILES string, with a column for every group of the method:

.. code-block:: python

   from fuellib.gcm.gani import decompose

   counts = decompose("CCc1ccccc1")
   counts.loc[:, (counts != 0).any()]

A :class:`ValueError` is raised if the SMILES cannot be parsed or an atom is
not matched to a first-order group. The same functionality is available from
the command line as ``fl decompose`` (see :doc:`tutorials_cli`). The
decompositions are used to add custom compounds (see
:doc:`tutorials_custom_fuels`).

.. warning::
   Automated decompositions must be verified. A known shortcoming of the
   ``gani`` decomposition is indene, where the non-aromatic C=C bond is
   categorized as a ``CH`` group.

.. _sec-gcm-gani:

Constantinou and Gani
---------------------

:mod:`fuellib.gcm.gani` implements the GCM of Constantinou and
Gani\ :footcite:p:`constantinou_new_1994,constantinou_estimation_1995`, with
additions from Govindaraju and Ihme\ :footcite:p:`govindaraju_group_2016`. It
predicts most of the compound properties at standard conditions. The
properties that can be measured (e.g., :attr:`~fuellib.fuel.components.Components.tc`)
use reference values from ``ref_compounds.csv`` when available and the
``gani`` predictions otherwise (see :ref:`sec-reference-properties`).

.. _tab-GCM-properties:

Table of Constantinou and Gani properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`tab-GCM-properties` outlines the properties of the *i-th* compound in a
mixture, which depend on the *k-th* first-order and *j-th* second-order group
contributions. Each property is registered as a ``gani`` property function
(e.g., :math:`T_{c,i}` as ``tc`` and :math:`\Delta\tilde{H}_{f,i}` as ``hf``).

.. table:: GCM properties of the *i-th* component in a mixture. The subscript *stp* denotes a standard pressure assumption.
   :widths: auto
   :align: center

   ============================================  =====================  ===========================================  ====================  ===========================================================
   Property                                      Units                  Group Contributions                          Units                 Description
   ============================================  =====================  ===========================================  ====================  ===========================================================
   :math:`T_{c,i}`                               K                      :math:`t_{c1k}`, :math:`t_{c2j}`             1                     Critical temperature\ :footcite:p:`constantinou_new_1994`.
   :math:`p_{c,i}`                               Pa                     :math:`p_{c1k}`, :math:`p_{c2j}`             bar\ :sup:`-0.5`      Critical pressure\ :footcite:p:`constantinou_new_1994`.
   :math:`V_{c,i}`                               m\ :sup:`3`\ /mol      :math:`v_{c1k}`, :math:`v_{c2j}`             m\ :sup:`3`\ /kmol    Critical volume\ :footcite:p:`constantinou_new_1994`.
   :math:`T_{b,i}`                               K                      :math:`t_{b1k}`, :math:`t_{b2j}`             1                     Normal boiling point\ :footcite:p:`constantinou_new_1994`.
   :math:`T_{m,i}`                               K                      :math:`t_{m1k}`, :math:`t_{m2j}`             1                     Normal melting point\ :footcite:p:`constantinou_new_1994`.
   :math:`\Delta\tilde{H}_{f,i}`                 J/mol                  :math:`h_{f1k}`, :math:`h_{f2j}`             kJ/mol                Enthalpy of formation at 298 K\ :footcite:p:`constantinou_new_1994`.
   :math:`\Delta\tilde{G}_{f,i}`                 J/mol                  :math:`g_{f1k}`, :math:`g_{f2j}`             kJ/mol                Standard Gibbs free energy at 298 K\ :footcite:p:`constantinou_new_1994`.
   :math:`\Delta\tilde{H}_{v,\textit{stp},i}`    J/mol                  :math:`h_{v1k}`, :math:`h_{v2j}`             kJ/mol                Enthalpy of vaporization at 298 K\ :footcite:p:`constantinou_new_1994`.
   :math:`\omega_i`                              1                      :math:`\omega_{1k}`, :math:`\omega_{2j}`     1                     Acentric factor\ :footcite:p:`constantinou_estimation_1995`.
   :math:`V_{m,\textit{stp},i}`                  m\ :sup:`3`\ /mol      :math:`v_{m1k}`, :math:`v_{m2j}`             m\ :sup:`3`\ /kmol    Liquid molar volume at 298 K\ :footcite:p:`constantinou_estimation_1995`.
   :math:`\tilde{C}_{p,i}`                       J/mol/K                :math:`C_{pA1_k}`, :math:`C_{pA2_k}`,...     J/mol/K               Molar specific heat capacity\ :footcite:p:`nielsen_molecular_1998` \ :footcite:p:`poling_properties_2001`.
   ============================================  =====================  ===========================================  ====================  ===========================================================

The ``gani`` method also registers the Ruzicka-Domalski heat capacity
coefficients ``rd_a``, ``rd_b``, and ``rd_d``\ :footcite:p:`ruzicka_estimation_1993`
and the flash point contributions ``alibakhshi_phi``\ :footcite:p:`alibakhshi_modified_2015`,
which are used by the :ref:`eq-GCM-correlations`.

.. _eq-GCM-properties:

Equations for Constantinou and Gani properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The properties of each compound in a mixture can be calculated as the sum of contributions
from the first- and second-order groups that make up the compound. For a given mixture,
let :math:`\mathbf{N}` be an :math:`N_c \times N_{g_1}` matrix that represents the
number of first-order groups in each compound, where :math:`N_c` is the number of compounds
in the mixture and :math:`N_{g_1}` is the total number of first-order groups as defined
by Constantinou and Gani\ :footcite:p:`constantinou_new_1994,constantinou_estimation_1995`.
Similarly, let :math:`\mathbf{M}` be an :math:`N_c \times N_{g_2}` matrix that specifies
the number of second-order groups in each compound, where :math:`N_{g_2}` is the total
number of second-order groups. The total number of groups :math:`N_g = N_{g_1} + N_{g_2} = 121`.
The GCM properties for the *i-th* compound in the mixture are calculated as
follows\ :footcite:p:`constantinou_new_1994` \ :footcite:p:`constantinou_estimation_1995` \ :footcite:p:`poling_properties_2001`:

.. math::

   T_{c,i} &= 181.28 \ln  \bigg[ \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} t_{c1k} + \sum_{j=1}^{N_{g_2}}         \mathbf{M}_{ij} t_{c2j} \bigg],\\
   p_{c,i} &= \Bigg( \bigg[  \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} p_{c1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij}     p_{c2j} + 0.10022\bigg]^{-2}  + 1.3705\Bigg)\times 10^{5}, \\
   V_{c,i} &= \Bigg( \bigg[ \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} v_{c1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij}      v_{c2j} \bigg] -0.00435 \Bigg)\times 10^{-3}, \\
   T_{b,i} &= 204.359 \ln  \bigg[ \sum_{k = 1}^{N_{g_1}} \mathbf{N}_{ik} t_{b1k} + \sum_{j=1}^{N_{g_2}}  \mathbf{M}_{ij} t_{b2j}\bigg],\\
   T_{m,i} &= 102.425 \ln  \bigg[ \sum_{k = 1}^{N_{g_1}} \mathbf{N}_{ik} t_{m1k} + \sum_{j=1}^{N_{g_2}}  \mathbf{M}_{ij} t_{m2j}\bigg],\\
   \Delta\tilde{H}_{f,i} &= \Bigg( \bigg[ \sum_{k = 1}^{N_{g_1}} \mathbf{N}_{ik} h_{f1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} h_{f2j} \bigg] + 10.835\Bigg) \times 10^3,\\
   \Delta\tilde{G}_{f,i} &= \Bigg( \bigg[ \sum_{k = 1}^{N_{g_1}} \mathbf{N}_{ik} g_{f1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} g_{f2j} \bigg] -14.828 \Bigg) \times 10^3,\\
   \Delta\tilde{H}_{v,\textit{stp},i} &= \Bigg( \bigg[ \sum_{k = 1}^{N_{g_1}} \mathbf{N}_{ik} h_{v1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} h_{v2j} \bigg] + 6.829\Bigg) \times 10^3, \\
   \omega_i &= 0.4085 \bigg[ \ln \Big(  \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} \omega_{1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} \omega_{2j} + 1.1507\Big) \bigg]^{1/0.5050}, \\
   V_{m,\textit{stp},i} &= \Bigg( \bigg[ \sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} v_{m1k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} v_{m2j} \bigg] + 0.01211 \Bigg)\times 10^{-3}, \\
   \tilde{C}_{p,i} & =\bigg[\sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} C_{pA1_k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} C_{pA2_j} -19.7779\bigg]  \\
       & +\bigg[\sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} C_{pB1_k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} C_{pB2_j} + 22.5981\bigg] \theta \\
       & +\bigg[\sum_{k=1}^{N_{g_1}} \mathbf{N}_{ik} C_{pC1_k} + \sum_{j=1}^{N_{g_2}} \mathbf{M}_{ij} C_{pC2_j} - 10.7983\bigg] \theta^2 \\
   \theta &= \frac{T - 298.15}{700}

The decomposition matrices :math:`\mathbf{N}` and :math:`\mathbf{M}` are read
from ``ref_gani.csv`` as ``Fuel._gani_decomp``, and the group contributions
(:math:`t_{c1k}`, :math:`t_{b1k}`, ...) are read from ``fuellib/gcm/gani.csv``.

.. _sec-gcm-nannoolal:

Nannoolal
---------

:mod:`fuellib.gcm.nannoolal` implements the saturated liquid viscosity GCM of
Nannoolal et al.\ :footcite:p:`nannoolal_estimation_2009`. The group
contributions are read from ``fuellib/gcm/nannoolal.csv``, which has one row
per first- or second-order group and columns for the group ``Order``, matching
``Priority``, and the ``dBv`` and ``Tv`` contributions. The decomposition
matrix :math:`\mathbf{N}` contains both first- and second-order groups and is
read from ``ref_nannoolal.csv`` as ``Fuel._nannoolal_decomp``.

.. _tab-nannoolal-properties:

Table of Nannoolal properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. table:: Nannoolal GCM properties of the *i-th* component in a mixture.
   :widths: auto
   :align: center

   ===========================  =====  ===============================  =====  ============================================================================
   Property                     Units  Group Contributions              Units  Description
   ===========================  =====  ===============================  =====  ============================================================================
   :math:`dBv_i`                1      :math:`\widetilde{dBv}_k`        1      Slope of the viscosity curve\ :footcite:p:`nannoolal_estimation_2009`.
   :math:`T_{v,i}`              K      :math:`\widetilde{T}_{v,k}`      1      Viscosity reference temperature\ :footcite:p:`nannoolal_estimation_2009`.
   ===========================  =====  ===============================  =====  ============================================================================

The ``nannoolal`` method registers the dimensionless group sums as ``dbv`` and
``tv``:

.. code-block:: python

   from fuellib.gcm import nannoolal_gcm

   nannoolal_gcm.predict("dbv", fuel)   # dimensionless Quantity1D
   nannoolal_gcm.predict("tv", fuel)    # dimensionless Quantity1D

.. _eq-nannoolal-properties:

Equations for Nannoolal properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tabulated group contributions are :math:`\widetilde{dBv}_k` and
:math:`\widetilde{T}_{v,k}`, whereas :math:`dBv_i` and :math:`T_{v,i}` are
compound parameters calculated from the group sums and the normal boiling
point:

.. math::
   dBv_i &= \frac{\sum_k \mathbf{N}_{ik} \widetilde{dBv}_k}
      {n_i^{-2.5635} + 0.0685} + 3.7777, \\
   T_{v,i} &= 21.8444\sqrt{T_{b,i}}
      + \frac{\left(\sum_k \mathbf{N}_{ik} \widetilde{T}_{v,k}\right)^{0.9315}}
      {n_i^{0.6577} + 4.9259} - 231.1361,

where :math:`n_i` is the number of non-hydrogen atoms of the compound. These
corrections are applied by
:func:`~fuellib.correlate.components.viscosity_dynamic_nannoolal`, which gives
the dynamic viscosity (see :ref:`sec-kinematic-viscosity`).

Some groups have no published viscosity contribution. Groups stored as
``0.0`` (``>C=C=C<``, ``-C#C-C#C-``, ``C2C-CC2``, and ``Ortho-pair(s)``)
trigger a :class:`UserWarning` listing the affected compounds. Groups with no
value (``C3C-CC2`` and ``C3C-CC3``) propagate ``NaN`` to the compounds that
contain them.

Nannoolal decomposition
^^^^^^^^^^^^^^^^^^^^^^^

:func:`fuellib.gcm.nannoolal.decompose` matches first-order groups with RDKit
SMARTS patterns in order of the Nannoolal priority, and each atom is assigned
to only one first-order group. Second-order groups are additive and are
counted from SMARTS matches and structural checks: the number of hydrogens
(``No hydrogen``, ``One hydrogen``), non-aromatic three-, four-, and
five-membered rings (``3/4 ring``, ``5 ring``), and ortho, meta, or para
substituent pairs on benzene rings (counted only when a single pair type is
present). The same functionality is available as
``fl decompose --method nannoolal``.

.. _sec-gcm-unifac:

UNIFAC
------

:mod:`fuellib.gcm.unifac` implements the UNIFAC 2.0 activity coefficient
model of Hayer et al.\ :footcite:p:`hayer_advancing_2025`. UNIFAC 2.0 keeps the
UNIFAC subgroup volume and surface parameters, :math:`R_k` and :math:`Q_k`, and
replaces the main-group interaction parameters :math:`a_{mn}` with a complete
table obtained with machine learning. The parameters are read from two files:

- ``fuellib/gcm/unifac.csv``: one column per subgroup with its subgroup number,
  name, main group number and name, and the ``R`` and ``Q`` parameters. The
  table has two subgroups named ``CHO``; the second is renamed ``CHO.1``.
- ``fuellib/gcm/unifac_amn.csv``: the interaction parameters :math:`a_{mn}`
  in K, with rows :math:`m` and columns :math:`n` indexed by main group number.

The decomposition matrix :math:`\boldsymbol{\nu}`, where :math:`\nu_{ki}` is
the number of subgroups :math:`k` in compound :math:`i`, is read from
``ref_unifac.csv`` as ``Fuel._unifac_decomp``.

.. _tab-unifac-properties:

Table of UNIFAC properties
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. table:: UNIFAC 2.0 properties of the *i-th* component in a mixture.
   :widths: auto
   :align: center

   ======================  =====  ======================  =====  ======================================================================
   Property                Units  Group Contributions     Units  Description
   ======================  =====  ======================  =====  ======================================================================
   :math:`r_i`             1      :math:`R_k`             1      Volume parameter\ :footcite:p:`hayer_advancing_2025`.
   :math:`q_i`             1      :math:`Q_k`             1      Surface parameter\ :footcite:p:`hayer_advancing_2025`.
   :math:`\ln \gamma_i`    1      :math:`a_{mn}`          K      Activity coefficient in the liquid mixture\ :footcite:p:`hayer_advancing_2025`.
   ======================  =====  ======================  =====  ======================================================================

The ``unifac`` method registers :math:`r_i` and :math:`q_i` as ``r`` and ``q``:

.. code-block:: python

   from fuellib.gcm import unifac_gcm

   unifac_gcm.predict("r", fuel)   # dimensionless Quantity1D
   unifac_gcm.predict("q", fuel)   # dimensionless Quantity1D

.. _eq-unifac-properties:

Equations for UNIFAC properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The volume and surface parameters of the *i-th* compound are

.. math::

   r_i = \sum_k \nu_{ki} R_k, \qquad q_i = \sum_k \nu_{ki} Q_k.

The activity coefficient of the *i-th* compound is the sum of a combinatorial
and a residual part, :math:`\ln \gamma_i = \ln \gamma_i^C + \ln \gamma_i^R`.
The combinatorial part accounts for differences in molecular size and shape:

.. math::

   \ln \gamma_i^C &= 1 - V_i + \ln V_i - 5 q_i \left( 1 - \frac{V_i}{F_i} + \ln \frac{V_i}{F_i} \right), \\
   V_i &= \frac{r_i}{\sum_j x_j r_j}, \qquad F_i = \frac{q_i}{\sum_j x_j q_j},

where :math:`x_j` are the liquid mole fractions. The residual part accounts for
the interactions between subgroups:

.. math::

   \ln \gamma_i^R &= \sum_k \nu_{ki} \left( \ln \Gamma_k - \ln \Gamma_k^{(i)} \right), \\
   \ln \Gamma_k &= Q_k \left[ 1 - \ln \left( \sum_m \Theta_m \Psi_{mk} \right) - \sum_m \frac{\Theta_m \Psi_{km}}{\sum_n \Theta_n \Psi_{nm}} \right], \\
   \Theta_m &= \frac{Q_m X_m}{\sum_n Q_n X_n}, \qquad X_m = \frac{\sum_j \nu_{mj} x_j}{\sum_j \sum_n \nu_{nj} x_j}, \qquad \Psi_{mn} = \exp \left( -\frac{a_{mn}}{T} \right),

where :math:`\Gamma_k` is the residual activity coefficient of subgroup
:math:`k` in the mixture and :math:`\Gamma_k^{(i)}` is the same quantity in pure
compound :math:`i`. The interaction parameter :math:`a_{mn}` of two subgroups
is that of their main groups, and :math:`a_{mm} = 0`. Subgroups of the same
main group (for example ``CH3`` and ``CH2``) therefore do not contribute to the
residual part.

:func:`fuellib.gcm.unifac.ln_gamma` evaluates :math:`\ln \gamma_i` for a fuel at
temperature ``temp``. The composition is given as liquid mass fractions ``yi``
(default: :attr:`~fuellib.fuel.core.Fuel.y0`) and converted to mole fractions.
For repeated evaluations, such as in a solver loop,
:func:`~fuellib.gcm.unifac.parameters` collects the arrays
:math:`\boldsymbol{\nu}`, :math:`R_k`, :math:`Q_k`, and :math:`a_{mn}` for the
subgroups present in the fuel once, and
:func:`~fuellib.gcm.unifac.ln_gamma_x` evaluates :math:`\ln \gamma_i` with
plain NumPy arrays from mole fractions and a temperature in K:

.. code-block:: python

   from fuellib import Fuel, Units
   from fuellib.gcm import unifac

   fuel = Fuel("posf10325")
   unifac.ln_gamma(fuel, Units.Quantity(300.0, "K"))   # dimensionless Quantity1D

   params = unifac.parameters(fuel)
   x = fuel.comps.y2x(fuel.y0).to("dimensionless").magnitude
   unifac.ln_gamma_x(x, 300.0, *params)                # NumPy array

UNIFAC decomposition
^^^^^^^^^^^^^^^^^^^^

:func:`fuellib.gcm.unifac.decompose` matches only the hydrocarbon subgroups,
in order of priority: ``ACCH3``, ``ACCH2``, ``ACCH``, ``ACH``, ``AC``,
``CH2=CH``, ``CH=CH``, ``CH2=C``, ``CH=C``, ``C=C``, ``CH3``, ``CH2``, ``CH``,
and ``C``. A :class:`ValueError` is raised if an atom is not matched to a
subgroup, for example a heteroatom. The same functionality is available as
``fl decompose --method unifac``.

.. _sec-gcm-boehm:

Boehm
-----

:mod:`fuellib.gcm.boehm` provides the entropy of fusion of each compound
following Boehm et al.\ :footcite:p:`boehm_blend_2022`, which is used by the
mixture freeze point (see :ref:`sec-mixture-freeze-point`). Rather than a
group decomposition, it uses the hydrocarbon family from the ``Family`` column
of ``ref_compounds.csv`` (available as ``fuel.components_data["Family"]``) and
the carbon number of each compound. The families are ``n-alkane``,
``isoalkane``, ``alkene``, ``monocycloalkane``, ``dicycloalkane``,
``tricycloalkane``, ``alkylbenzene``, ``cycloaromatic``, and ``diaromatic``.

.. _tab-boehm-properties:

Table of Boehm properties
^^^^^^^^^^^^^^^^^^^^^^^^^

.. table:: Boehm properties of the *i-th* component in a mixture.
   :widths: auto
   :align: center

   ===================================  =======  ==============================================================  =======  ========================================================
   Property                             Units    Family Parameters                                               Units    Description
   ===================================  =======  ==============================================================  =======  ========================================================
   :math:`\Delta S_{\textit{fus},i}`    J/mol/K  :math:`A_f`, :math:`B_f`, :math:`C_{\textit{ref},f}`            J/mol/K  Entropy of fusion\ :footcite:p:`boehm_blend_2022`.
   ===================================  =======  ==============================================================  =======  ========================================================

The ``boehm`` method registers :math:`\Delta S_{\textit{fus},i}` as ``ds_fus``:

.. code-block:: python

   from fuellib.gcm import boehm_gcm

   boehm_gcm.predict("ds_fus", fuel)   # Quantity1D in J/(mol*K)

.. _eq-boehm-properties:

Equations for Boehm properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The entropy of fusion is a linear function of the carbon number
:math:`n_{C,i}`, with parameters :math:`A_f`, :math:`B_f`, and
:math:`C_{\textit{ref},f}` for family :math:`f` read from
``fuellib/gcm/boehm.csv``:

.. math::

   \Delta S_{\textit{fus},i} = \max\big[A_f + B_f (n_{C,i} - C_{\textit{ref},f}),\, 20\big].

Compounds whose family is not in the table fall back to the Walden-rule value
of 56.5 J/mol/K.

References
----------

.. footbibliography::
