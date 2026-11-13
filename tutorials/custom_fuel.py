import fuellib as fl

# Load an embedded fuel
fuel = fl.Fuel("posf10264")

print(f"Fuel: {fuel.name}")
print(f"Fuel data directory: {fuel.data_dir}")
print(f"Properties data available: {fuel.prop_data is not None}")
print(f"Number of compounds: {fuel.num_compounds}")

# To use a custom fuel, create a directory structure like:
# customFuels/fuelData/
#   ├── gc_data/
#   │   └── myFuel.csv
#   ├── properties_data/  (optional)
#   │   └── myFuel.csv
#   ├── ref_compounds.csv  (optional, defaults to fuellib/data/ref_compounds.csv)
#   └── ref_gani.csv  (optional, defaults to fuellib/data/ref_gani.csv)
#
# Then load it with:
custom_fuel = fl.Fuel("myFuel", data_dir="customFuels/fuelData")

# After loading, the fuel object has the correct directory path:
custom_fuel.data_dir

print(f"\nFuel: {custom_fuel.name}")
print(f"Fuel data directory: {custom_fuel.data_dir}")
print(f"Properties data available: {custom_fuel.prop_data is not None}")
print(f"Number of compounds: {custom_fuel.num_compounds}")
