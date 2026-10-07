from pathlib import Path

import fuellib as fl

# Load a fuel from the FuelLib database
fuel = fl.Fuel("posf10264")

print(f"Fuel: {fuel.name}")
print(f"User database directory: {fuel.userDataDir}")
print(f"Number of compounds: {fuel.num_compounds}")

# To use a custom fuel, create a user database mirroring fuellib/database:
# customFuels/
#   ├── gcData/
#   │   └── myFuel.csv            (Common_Name and/or SMILES, Weight %)
#   ├── referenceCompounds/        (optional: compounds not in FuelLib)
#   │   ├── compounds.csv
#   │   └── gani.csv
#   └── propertiesData/            (optional)
#       └── myFuel.csv
#
# A blank template with the CSV headers filled in can be created with:
if not Path("customFuels").exists():
    fl.database.write_template("customFuels", "myFuel")

# After filling in the template, load the fuel with:
# custom_fuel = fl.Fuel("myFuel", userDataDir="customFuels")
#
# Missing Family, Num_C, InChI, and GCM-predictable properties in
# customFuels/referenceCompounds/compounds.csv are populated automatically
# (and written back) when the fuel is loaded.
