"""Core Group Contribution Method registry."""

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel
from rdkit import Chem

from ..utils import types

if TYPE_CHECKING:
    from ..fuel import Fuel


__all__ = ["Decomposer", "GCMRegistry"]


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


class Decomposer:
    """Decompose molecules into groups for a Group Contribution Method.

    Subclasses with second order groups should override `second_order_counts`.
    """

    def __init__(
        self,
        method: str,
        table_groups: pd.Index,
        first_order_groups: Sequence[tuple[str, str]],
    ) -> None:
        """Initialize the decomposer with a GCM's group data.

        Args:
            method: Display name of the GCM used in error messages.
            table_groups: Group names that are columns of the GCM parameter table.
            first_order_groups: ``(group name, SMARTS)`` pairs ordered by priority.
        """
        self.method = method
        self.table_groups = table_groups
        self.first_order_groups = first_order_groups

    def __call__(self, smiles: str) -> pd.DataFrame:
        """Decompose a molecule into groups.

        Args:
            smiles: The SMILES string of the molecule to decompose.

        Returns:
            A single-row DataFrame indexed by the SMILES string, with a column for every
            group in the GCM table containing the number of occurrences of that group.

        Raises:
            KeyError: If a matched group is not a column of the GCM table.
        """
        mol = self.parse_smiles(smiles)

        counts = Counter(self.first_order_counts(mol, smiles))
        counts.update(self.second_order_counts(mol, smiles))

        invalid = set(counts) - set(self.table_groups)
        if invalid:
            msg = (
                f"{self.context(smiles)}: groups not found in {self.method} table: "
                f"{sorted(invalid)}"
            )
            raise KeyError(msg)

        df = pd.DataFrame(0, index=[smiles], columns=self.table_groups)
        for group, count in counts.items():
            df.loc[smiles, group] += count
        return df

    def context(self, smiles: str) -> str:
        """Build the shared prefix of every decomposition error message.

        Args:
            smiles: The SMILES string being decomposed.

        Returns:
            The message prefix naming the method and SMILES string.
        """
        return f"{self.method} decomposition failed for SMILES {smiles!r}"

    def parse_smiles(self, smiles: str) -> Chem.Mol:
        """Validate and parse a SMILES string.

        Non-string values, such as the NaN pandas produces for a blank CSV cell, are
        rejected, as are empty strings (RDKit would parse those as an empty molecule).

        Args:
            smiles: The SMILES string to parse.

        Returns:
            The parsed RDKit molecule.

        Raises:
            ValueError: If the SMILES is not a non-empty string or cannot be parsed.
        """
        if not isinstance(smiles, str) or not smiles.strip():
            msg = (
                f"{self.method} decomposition failed: "
                "SMILES must be a non-empty string."
            )
            raise ValueError(msg)
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            msg = f"{self.context(smiles)}: RDKit could not parse the input."
            raise ValueError(msg)
        return mol

    def parse_smarts(self, smiles: str, smarts: str) -> Chem.Mol:
        """Parse a SMARTS pattern used to decompose a molecule.

        Args:
            smiles: The SMILES string being decomposed, used in error messages.
            smarts: The SMARTS pattern to parse.

        Returns:
            The RDKit query molecule.

        Raises:
            ValueError: If the SMARTS pattern is invalid.
        """
        query = Chem.MolFromSmarts(smarts)
        if not query:
            msg = f"{self.context(smiles)}: invalid SMARTS pattern {smarts!r}."
            raise ValueError(msg)
        return query

    def first_order_counts(self, mol: Chem.Mol, smiles: str) -> dict[str, int]:
        """Count first order groups, requiring that they cover every atom.

        Args:
            mol: The parsed molecule.
            smiles: The SMILES string being decomposed, used in error messages.

        Returns:
            A mapping of first order group name to its number of occurrences.

        Raises:
            ValueError: If a SMARTS pattern is invalid or any atom is not matched to a
                first order group.
        """
        matched: set[int] = set()
        counts: dict[str, int] = {}
        for group, smarts in self.first_order_groups:
            query = self.parse_smarts(smiles, smarts)
            for match in mol.GetSubstructMatches(query):
                if matched.intersection(match):
                    continue
                matched.update(match)
                counts[group] = counts.get(group, 0) + 1

        atom_count = mol.GetNumAtoms()
        if len(matched) != atom_count:
            msg = (
                f"{self.context(smiles)}: first-order groups matched "
                f"{len(matched)} of {atom_count} atoms."
            )
            raise ValueError(msg)
        return counts

    def second_order_counts(self, mol: Chem.Mol, smiles: str) -> dict[str, int]:
        """Count second order groups.

        Override in GCMs that define second order groups. The default has none.

        Args:
            mol: The parsed molecule.
            smiles: The SMILES string being decomposed, used in error messages.

        Returns:
            A mapping of second order group name to its number of occurrences.
        """
        return {}
