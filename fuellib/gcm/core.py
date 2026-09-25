"""Core Group Contribution Method registry."""

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel

from ..utils import types

if TYPE_CHECKING:
    from ..fuel import Fuel


__all__ = ["GCMRegistry"]


@runtime_checkable
class PropertyProtocol(Protocol):
    """Protocol for properties that can be predicted by GCMs."""

    __name__: str

    def __call__(self, fuel: "Fuel") -> types.Quantity1D:
        """Predict the property for the given fuel's components."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class GCM(BaseModel):
    """Base class for Core Group Contribution Methods."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    property_fns: dict[str, PropertyProtocol]

    def register_property(self, function: PropertyProtocol) -> PropertyProtocol:
        """Register a new property implementation for this GCM.

        Args:
            function: The property function to register.

        Returns:
            The registered property function.
        """
        self.property_fns[function.__name__.lower()] = function
        return function

    def list_property_fns(self) -> list[str]:
        """List the names of all registered property functions.

        Returns:
            A list of the names of all registered property functions.
        """
        return sorted(self.property_fns.keys())

    def get_property(self, property_name: str) -> PropertyProtocol:
        """Retrieve the property function for the specified property name.

        Args:
            property_name: The name of the property function to retrieve.

        Returns:
            The property function corresponding to the specified property name.

        Raises:
            ValueError: If the specified property is not implemented by this GCM.
        """
        property_name = property_name.lower()
        prop = self.property_fns.get(property_name)
        if prop is not None:
            return prop
        msg = (
            f"Property '{property_name}' is not implemented by {self.name} GCM. "
            f"Available property functions: {self.list_property_fns()}"
        )
        raise ValueError(msg)

    def predict(self, property_name: str, fuel: "Fuel") -> types.Quantity1D:
        """Predict the specified property for a fuel's components.

        Args:
            property_name: The name of the property to predict.
            fuel: The fuel object containing the decomposition matrix.

        Returns:
            The predicted property values for the fuel's components as a Quantity1D.
        """
        property = self.get_property(property_name)
        return property(fuel)

    def predict_all(self, fuel: "Fuel") -> dict[str, dict[str, types.Quantity1D]]:
        """Predict all registered property functions for a fuel's components.

        Args:
            fuel: The fuel object containing the decomposition matrix.

        Returns:
            A dictionary mapping property names to their predicted values for the fuel's
                components.
        """
        return {
            self.name: {
                prop: self.predict(prop, fuel) for prop in self.list_property_fns()
            }
        }


class GCMRegistry:
    """Registry for Core Group Contribution Methods."""

    methods: ClassVar[list[GCM]] = []

    @classmethod
    def register(
        cls, gcm_name: str, property_fns: list[PropertyProtocol] | None = None
    ) -> GCM:
        """Register a new GCM with the given name and properties.

        Args:
            gcm_name: The name of the GCM to register.
            property_fns: A list of property functions to associate with the GCM.

        Returns:
            The registered GCM instance.
        """
        gcm = GCM.model_validate({
            "name": gcm_name.lower(),
            "property_fns": {prop.__name__.lower(): prop for prop in property_fns}
            if property_fns is not None
            else {},
        })
        cls.methods.append(gcm)
        return gcm

    @classmethod
    def list_methods(cls) -> list[str]:
        """List the names of all registered GCMs.

        Returns:
            A list of names of all registered GCMs.
        """
        return [gcm.name for gcm in cls.methods]

    @classmethod
    def get_gcm(cls, gcm_name: str) -> GCM:
        """Retrieve a registered GCM by its name.

        Args:
            gcm_name: The name of the GCM to retrieve.

        Returns:
            The registered GCM instance.

        Raises:
            ValueError: If no GCM with the given name is registered.
        """
        gcm_name = gcm_name.lower()
        for gcm in cls.methods:
            if gcm.name == gcm_name:
                return gcm
        msg = (
            f"GCM with name '{gcm_name}' is not registered. Available GCMs: "
            f"{cls.list_methods()}"
        )
        raise ValueError(msg)
