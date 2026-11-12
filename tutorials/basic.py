import fuellib as fl

# Create a fuel object for the fuel "heptane-decane"
fuel = fl.Fuel("heptane-decane")

# Display fuel name, components, initial composition, and critical temperature
print(f"Fuel name: {fuel.name}")
print(f"Fuel components: {fuel.compounds}")
print(f"Initial composition: {fuel.Y_0:~P.4f}")
print(f"Critical temperature: {fuel.Tc:~P.2f}")  # ~P for "pretty" pint unit formatting

# Calculate the saturated vapor pressure at 320 K
T = fl.Units.Quantity(320, "K")  # Temperature as a pint.Quantity
p_sat_i = fuel.psat(T)
p_vap_mix = fuel.mixture_vapor_pressure(fuel.Y_0, T)
print(f"Saturated vapor pressure at {T:~P}: {p_sat_i:~P.2f}")
print(f"Mixture vapor pressure at {T:~P}: {p_vap_mix:~P.2f}")
