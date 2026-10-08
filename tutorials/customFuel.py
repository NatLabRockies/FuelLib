import fuellib as fl

# Load an embedded fuel
fuel = fl.Fuel("posf10264")

print(f"Fuel: {fuel.name}")
print(f"Fuel data directory: {fuel.fuelDataDir}")
print(f"Properties data available: {fuel.propData is not None}")
print(f"Number of compounds: {fuel.num_compounds}")

# To use a custom fuel, create a directory structure like:
# customFuels/fuelData/
#   ├── gcData/
#   │   └── myFuel.csv
#   ├── propertiesData/  (optional)
#   │   └── myFuel.csv
#   ├── refCompounds.csv  (optional, defaults to fuellib/data/refCompounds.csv)
#   └── refGani.csv  (optional, defaults to fuellib/data/refGani.csv)
#
# Then load it with:
custom_fuel = fl.Fuel("hefa-S1", fuelDataDir="customFuels/fuelData")

# After loading, the fuel object has the correct directory path:
custom_fuel.fuelDataDir

print(f"\nFuel: {custom_fuel.name}")
print(f"Fuel data directory: {custom_fuel.fuelDataDir}")
print(f"Properties data available: {custom_fuel.propData is not None}")
print(f"Number of compounds: {custom_fuel.num_compounds}")
