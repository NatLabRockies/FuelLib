## References for fuel data

# GCxGC Data
The provided GC x GC data is for the weight % of the fuel.  

## POSF Fuels
The GCxGC data for the POSF fuels comes from the [National Jet Fuels Combustion Program](https://doi.org/10.2514/1.J055361) (NJFCP) as provided by the Air Force Research Laboratory in [Edwards (2020)](https://apps.dtic.mil/sti/pdfs/AD1093317.pdf) and [Edwards (2017)](https://doi.org/10.2514/6.2017-0146):
* gcData/posf4658_init.csv (from Appendix D [Edwards (2017)](https://doi.org/10.2514/6.2017-0146))
* gcData/posf10264_init.csv
* gcData/posf10289_init.csv
* gcData/posf10325_init.csv
* gcData/posf11498_init.csv

## HEFA:Jet-A Blends
GCxGC data for HEFA fuels produced from diﬀerent feedstocks (camelina, tallow, and mixed fat) courtesy of [Vozka et al. (2018)](https://doi.org/10.1021/acs.energyfuels.8b02787). Note that these can be blended with Jet-A to reproduce the density and viscosity measurement data from that paper:
* gcData/hefa-came.csv
* gcData/hefa-tall.csv
* gcData/hefa-mfat.csv
* gcData/jet-a.csv

# Group Decomposition Data

## POSF Fuels
The decompositions of each compound into its functional groups for the POSF fuels originated from [Govindaraju & Ihme (2016)](https://doi.org/10.1016/j.ijheatmasstransfer.2016.06.079), but have been updated for additional accuracy and to include the specific reference compounds used:
* groupDecompositionData/posf4658.csv
* groupDecompositionData/posf10264.csv
* groupDecompositionData/posf10289.csv
* groupDecompositionData/posf10325.csv
* groupDecompositionData/posf11498.csv

## HEFA:Jet-A Blends
The group decompositions of the HEFA and Jet-A from [Vozka et al. (2018)](https://doi.org/10.1021/acs.energyfuels.8b02787) have been derived from the `posf10325` decomposition:
* groupDecompositionData/hefa.csv
* groupDecompositionData/jet-a.csv

# Fuel Properties Data
## Single Component Fuels
The properties data for the single component fuels is from [NIST WebBook](https://webbook.nist.gov/chemistry/):
* propertiesData/decane-NIST.csv
* propertiesData/dodecane-NIST.csv
* propertiesData/heptane-NIST.csv

## POSF Fuels
The properties data for POSF fuels is from the AFRL [Edwards (2017)](https://doi.org/10.2514/6.2017-0146) and [Edwards (2020)](https://apps.dtic.mil/sti/pdfs/AD1093317.pdf):
* propertiesData/posf10264.csv
* propertiesData/posf10289.csv
* propertiesData/posf10325.csv
* propertiesData/posf11498.csv

## HEFA:Jet-A Blends
The properties data for the HEFA:Jet-A blends are from [Vozka et al. (2018)](https://doi.org/10.1021/acs.energyfuels.8b02787) 
* propertiesData/hefa-jet-a-blends.csv