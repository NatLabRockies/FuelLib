import fuellib as fl

# Create a fuel object for the fuel "heptane-decane"
fuel = fl.Fuel("heptane-decane")

# Display fuel name, components, initial composition, and critical temperature
print(f"Fuel name: {fuel.name}")
print(f"Fuel components: {fuel.compounds}")
print(f"Initial composition: {fuel.Y_0}")
print(f"Critical temperature: {fuel.Tc}")

# Calculate the saturated vapor pressure at 320 K
T = fl.Units.Quantity(320, "K")  # Temperature as a pint.Quantity
p_sat_i = fuel.psat(T)
p_sat_mix = fuel.mixture_vapor_pressure(fuel.Y_0, T)
print(f"Saturated vapor pressure at {T} K: {p_sat_i}")
print(f"Mixture saturated vapor pressure at {T} K: {p_sat_mix:.2f}")
