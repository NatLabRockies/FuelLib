Welcome to FuelLib's documentation!
===================================

The **Fuel Library (FuelLib)** utilizes
the group contribution method (GCM), as developed by Constantinou and 
Gani\ :footcite:p:`constantinou_new_1994` \ :footcite:p:`constantinou_estimation_1995` in the mid-1990s
with additions from Govindaraju and Ihme (2016)\ :footcite:p:`govindaraju_group_2016`, 
to provide a systematic approach for estimating the thermodynamic properties of
pure organic compounds and mixtures of organic compounds. If you need help or have questions, please use the 
`GitHub discussion <https://github.com/NatLabRockies/FuelLib/discussions>`_.
The source code is available at `github.com/NatLabRockies/FuelLib <https://github.com/NatLabRockies/FuelLib>`_.

.. figure:: /figures/info-graphic.png
   :width: 600pt
   :align: center


Citing this work
----------------

If you use FuelLib in your research, please cite the following software record:

.. code-block:: none

   Montgomery, David, Appukuttan, Sreejith, Yellapantula, Shashank, Perry, Bruce, and Binswanger, Adam. FuelLib (Fuel Library) [SWR-25-26]. Computer Software. https://github.com/NatLabRockies/FuelLib. USDOE Office of Energy Efficiency and Renewable Energy (EERE), Office of Sustainable Transportation. Vehicle Technologies Office (VTO). 27 Feb. 2025. Web. doi:10.11578/dc.20250317.1.

.. code-block:: bibtex

   @misc{montgomery_fuellib_2025,
      title = {FuelLib (Fuel Library) [SWR-25-26]},
      author = {Montgomery, David and Appukuttan, Sreejith and Yellapantula, Shashank and Perry, Bruce and Binswanger, Adam},
      abstractNote = {FuelLib is a library that utilizes the group contribution method (GCM) for calculating thermodynamic properties of hydro-carbon jet fuels. FuelLib utilizes the tables and functions of the GCM as proposed by Constantinou and Gani (1994) and Constantinou, Gani and O'Connel (1995), with additional physical properties discussed in Govindaraju & Ihme (2016). The code is based on Pavan B. Govindaraju's Matlab implementation of the GCM, and has been expanded to include additional thermodynamic properties and mixture properties. The fuel library contains gas chromatography (GC x GC) data for a variety of fuels ranging from simple single component fuels to complex jet fuels. The GC x GC data for POSF jet fuels comes from Edwards (2020).},
      doi = {10.11578/dc.20250317.1},
      url = {https://doi.org/10.11578/dc.20250317.1},
      howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20250317.1}},
      year = {2025},
      month = {feb}
   }

.. toctree::
   :maxdepth: 4
   :includehidden:
   :caption: Contents:

   fuelprops
   sourcecode
   tutorials

.. footbibliography::