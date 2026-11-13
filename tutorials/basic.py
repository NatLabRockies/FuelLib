import fuellib as fl

# Create a fuel object for the fuel "heptane_decane"
fuel = fl.Fuel("heptane_decane")

# Display fuel name, components, initial composition, and critical temperature
print(f"Fuel name: {fuel.name}")
print(f"Fuel components: {fuel.comps.names}")
print(f"Initial composition: {fuel.y0:~P.4f}")
# ~P for "pretty" pint unit formatting
print(f"Critical temperature: {fuel.comps.tc:~P.2f}")

# Calculate the saturated vapor pressure at 320 K
T = fl.Units.Quantity(320, "K")  # Temperature as a pint.Quantity
p_sat_i = fuel.comps.pvap(T)
p_vap_mix = fuel.mix.pvap(fuel.y0, T)
print(f"Saturated vapor pressure at {T:~P}: {p_sat_i:~P.2f}")
print(f"Mixture vapor pressure at {T:~P}: {p_vap_mix:~P.2f}")
