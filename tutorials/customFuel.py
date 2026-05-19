import fuellib as fl

# Load an embedded fuel
fuel = fl.fuel("posf10264")

print(f"Fuel: {fuel.name}")
print(f"Fuel data directory: {fuel.fuelDataDir}")
print(f"GC data directory: {fuel.fuelDataGcDir}")
print(f"Decomposition directory: {fuel.fuelDataDecompDir}")
print(f"Properties directory: {fuel.fuelDataPropsDir}")
print(f"Number of compounds: {fuel.num_compounds}")

# To use a custom fuel, create a directory structure like:
# customFuels/fuelData/
#   ├── gcData/
#   │   └── myFuel_init.csv
#   ├── groupDecompositionData/
#   │   └── myFuel.csv
#   └── propertiesData/  (optional)
#       └── myFuel.csv
#
# Then load it with:
custom_fuel = fl.fuel("hefa-S1", fuelDataDir="customFuels/fuelData")

# After loading, the fuel object has the correct directory paths:
custom_fuel.fuelDataDir
custom_fuel.fuelDataGcDir
custom_fuel.fuelDataDecompDir
custom_fuel.fuelDataPropsDir

print(f"\nFuel: {custom_fuel.name}")
print(f"Fuel data directory: {custom_fuel.fuelDataDir}")
print(f"GC data directory: {custom_fuel.fuelDataGcDir}")
print(f"Decomposition directory: {custom_fuel.fuelDataDecompDir}")
print(f"Properties directory: {custom_fuel.fuelDataPropsDir}")
print(f"Number of compounds: {custom_fuel.num_compounds}")
