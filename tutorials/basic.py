import fuellib as fl

# Create a fuel object for the fuel "heptane-decane"
fuel = fl.Fuel("heptane-decane")

# Display fuel name, components, initial composition, and critical temperature
print(f"Fuel name: {fuel.name}")
print(f"Fuel components: {fuel.compounds}")
print(f"Initial composition: {fuel.Y_0}")
# Critical / correlated properties will print with units
print(f"Critical temperature ( K): {fuel.Tc:.2f}")
# We can also convert the units with the `.to()` method
print(f"Critical temperature (°C): {fuel.Tc.to('°C'):.2f}\n")

# Calculate the saturated vapor pressure at 320 K
T = fl.PintUnits.Quantity(320.0, "K") # Assign units to the temperature value
p_sat_i = fuel.psat(T)
p_sat_mix = fuel.mixture_vapor_pressure(fuel.Y_0, T)
print(f"Saturated vapor pressure at {T}: {p_sat_i:.2f}")
print(f"Mixture saturated vapor pressure at {T}: {p_sat_mix:.2f}")
