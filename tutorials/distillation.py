import matplotlib.pyplot as plt
import numpy as np

import fuellib as fl
from fuellib.simulate.distillation.greenfield import D86Apparatus, distill

# Create a fuel object for the fuel "POSF10325" (Jet A-3)
fuel = fl.Fuel("posf10325")  # Jet A-3

# Simulate the ASTM D86 distillation test with the full model
result = distill(fuel)

# D86 temperature at 10%, 50%, and 90% volume recovered
T10, T50, T90 = result.temperature_at(np.array([0.1, 0.5, 0.9])).to("degC")
print(f"T10: {T10:~P.1f}, T50: {T50:~P.1f}, T90: {T90:~P.1f}")

# Variants of the model
partial = distill(fuel, air_cstr=False)  # no air-filled CSTR
ideal = distill(fuel, ideal_solution=True, ideal_vapor=True)  # ideal mixture and vapor

# Different apparatus parameters
apparatus = D86Apparatus(air_volume=fl.Units.Quantity(100.0, "cm^3"))
large_air_volume = distill(fuel, apparatus=apparatus)

# Plot and compare the D86 curves (volume % recovered vs. D86 temperature), with the
# measured D86 data of the fuel shown as symbols
ax = result.plot(
    label="Full model",
    temperature_units="degC",
    data=fl.plotting.d86_data(fuel),
    data_label="Data",
)
partial.plot(ax=ax, label="No CSTR", temperature_units="degC")
ideal.plot(ax=ax, label="Ideal", temperature_units="degC", linestyle="--")
large_air_volume.plot(
    ax=ax, label="100 cm$^3$ air", temperature_units="degC", linestyle=":"
)

plt.show()
