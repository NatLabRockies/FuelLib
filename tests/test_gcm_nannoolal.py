"""Tests for the Nannoolal liquid-viscosity group-contribution method."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import fuellib as fl
from fuellib.fuel import Fuel
from fuellib.gcm import GCMRegistry, decompose, nannoolal_gcm
from fuellib.gcm.nannoolal import dynamic_viscosity, parameters
from fuellib.gcm.nannoolalDecomposition import map_gani_to_nannoolal_groups
from fuellib.utils import Units


def test_nannoolal_is_registered_with_viscosity_parameters():
    """Nannoolal supplies the dBv and Tv parameters through the GCM registry."""
    assert GCMRegistry.get_gcm("NANNOOLAL") is nannoolal_gcm
    assert nannoolal_gcm.list_property_fns() == ["dbv", "tv"]


def test_nannoolal_mapper_preserves_hydrocarbon_carbon_count():
    """The mapper translates Gani hydrocarbon groups without losing carbon atoms."""
    mapped, atom_count = map_gani_to_nannoolal_groups({"CH3": 2, "CH2": 10})
    assert mapped == {1: 2, 4: 10}
    assert atom_count == 12


def test_nannoolal_smiles_dispatcher_uses_gani_decomposition():
    """The generic dispatcher accepts Nannoolal and returns its group IDs."""
    assert decompose("CCCCCCC", "nannoolal") == {"1": 2, "4": 5}


def test_nannoolal_equation_reference_value():
    """The Nannoolal temperature equation returns the reference viscosity in Pa*s."""
    result = dynamic_viscosity(298.15, np.array([4.0]), np.array([300.0]))
    expected = 1.3e-3 * np.exp(-4.0 * (298.15 - 300.0) / (298.15 - 300.0 / 16.0))
    np.testing.assert_allclose(result, [expected])


def test_nannoolal_component_and_mixture_viscosities():
    """Selected Nannoolal component values flow through both mixture rules."""
    fuel = fl.Fuel("heptane-decane")
    temperature = Units.Quantity(298.15, "K")
    dynamic = fuel.viscosity_dynamic(temperature, model="Nannoolal")
    kinematic = fuel.viscosity_kinematic(temperature, model="nannoolal")

    assert np.isfinite(dynamic.magnitude).all() and (dynamic.magnitude > 0).all()
    np.testing.assert_allclose(
        dynamic.to("Pa*s").magnitude,
        (kinematic * fuel.density(temperature)).to("Pa*s").magnitude,
    )
    mixture = fuel.mixture_kinematic_viscosity(fuel.Y_0, temperature, model="Nannoolal")
    assert mixture.magnitude > 0


def test_hybrid_selection_keeps_dutt_for_normal_alkanes():
    """Hybrid retains Dutt for normal alkanes while strict Nannoolal evaluates them."""
    fuel = fl.Fuel("heptane-decane")
    temperature = Units.Quantity(298.15, "K")

    assert not np.allclose(
        fuel.viscosity_dynamic(temperature, model="Nannoolal").magnitude,
        fuel.viscosity_dynamic(temperature, model="Dutt").magnitude,
    )
    np.testing.assert_allclose(
        fuel.viscosity_dynamic(temperature, model="Hybrid").magnitude,
        fuel.viscosity_dynamic(temperature, model="Dutt").magnitude,
    )
    np.testing.assert_allclose(
        fuel.mixture_kinematic_viscosity(
            fuel.Y_0, temperature, model="Hybrid"
        ).magnitude,
        fuel.mixture_kinematic_viscosity(fuel.Y_0, temperature, model="Dutt").magnitude,
    )


def test_nannoolal_requires_valid_decompositions():
    """Nannoolal raises rather than silently falling back for invalid components."""
    invalid_fuel = SimpleNamespace(
        Tb=Units.Quantity([373.15], "K"),
        num_compounds=1,
        compounds=["water"],
        smiles=["O"],
    )
    with pytest.raises(ValueError, match="water"):
        parameters(cast(Fuel, invalid_fuel))


@pytest.mark.parametrize("model", ["unknown", "unsupported"])
def test_unknown_viscosity_model_is_rejected(model):
    """Viscosity entry points reject models outside the supported choices."""
    fuel = fl.Fuel("decane")
    with pytest.raises(ValueError, match="Unknown viscosity model"):
        fuel.viscosity_dynamic(Units.Quantity(298.15, "K"), model=model)
