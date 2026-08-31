import fuellib as fl
from fuellib.units import Q_

# Create a fuel object for the fuel "heptane-decane"
fuel = fl.fuel("heptane-decane")

# Display fuel name, components, initial composition, and critical temperature
print(f"Fuel name: {fuel.name}")
print(f"Fuel components: {fuel.compounds}")
print(f"Initial composition: {fuel.Y_0}")
print(f"Critical temperature: {fuel.Tc}")

# Calculate the saturated vapor pressure at 320 K
T = Q_(320.0, "K")
p_sat_i = fuel.psat(T)
p_sat_mix = fuel.mixture_vapor_pressure(fuel.Y_0, T)
print(f"Saturated vapor pressure at {T}: {p_sat_i}")
print(f"Mixture saturated vapor pressure at {T}: {p_sat_mix.to('Pa'):.2f}")
