Welcome to FuelLib's documentation!
===================================

The **Fuel Library (FuelLib)** is a fuel property prediction package based on
group contribution methods (GCMs) and, when available, measurement data.
FuelLib represents a fuel as a mixture of individual compounds from its GCxGC
composition. The properties of each compound are taken from measured reference
data or predicted with GCMs such as those of Constantinou and
Gani\ :footcite:p:`constantinou_new_1994,constantinou_estimation_1995`,
Nannoolal et al.\ :footcite:p:`nannoolal_estimation_2009`, and UNIFAC
2.0\ :footcite:p:`hayer_advancing_2025`. Correlations and mixture rules then give
the properties of the compounds and the fuel at other conditions (see
:doc:`prediction_framework`). If you need help or have questions, please use the
`GitHub discussion <https://github.com/NatLabRockies/FuelLib/discussions>`_.
The source code is available at `github.com/NatLabRockies/FuelLib <https://github.com/NatLabRockies/FuelLib>`_.

.. figure:: /figures/info-graphic.png
   :width: 600pt
   :align: center

.. include:: _includes/getting_started.inc

.. toctree::
   :maxdepth: 4
   :includehidden:
   :caption: Contents:

   getting_started
   prediction_framework
   group_contribution_methods
   reference_compounds
   correlations
   simulation
   tutorials
   validation
   source_code
   contributing

.. footbibliography::