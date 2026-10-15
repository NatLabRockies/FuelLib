"""Molecular group-decomposition algorithms and data utilities for GCM methods."""

import csv
from pathlib import Path

from .gani_decomposition import UnsupportedGroupError, decompose_gani


def decompose(smiles: str, method: str) -> dict[str, int]:
    """Decompose a SMILES string into group counts for a GCM method.

    Args:
        smiles: SMILES string for the molecule to decompose.
        method: Name of the group-contribution method.

    Returns:
        Mapping of method-specific group names to non-zero counts.

    Raises:
        ValueError: If the method is unsupported or the SMILES is invalid.
        UnsupportedGroupError: If the molecule is outside a method's supported scope.
    """
    if method.lower() == "gani":
        return decompose_gani(smiles)
    raise ValueError(f"Unsupported group-contribution method: {method!r}")


def group_names(method: str) -> list[str]:
    """Return the ordered group names for a group-contribution method.

    Args:
        method: Name of the group-contribution method.

    Raises:
        ValueError: If the method is unsupported.
    """
    if method.lower() == "gani":
        from .gani import TABLE

        return TABLE.columns.tolist()
    raise ValueError(f"Unsupported group-contribution method: {method!r}")


def format_decomposition(counts: dict[str, int], method: str) -> str:
    """Format non-zero group counts in a method's group-column order."""
    return ", ".join(
        f"({counts[group]}x) {group}"
        for group in group_names(method)
        if counts.get(group, 0)
    ) or "No groups assigned"


def write_decompositions_csv(
    output_path: Path,
    decompositions: list[tuple[str, str, dict[str, int]]],
    method: str,
) -> None:
    """Write group decomposition counts using a method's reference CSV layout.

    Args:
        output_path: Destination CSV path.
        decompositions: Tuples of compound name, SMILES, and group counts. The
            SMILES value is retained for traceability during processing but is not
            written.
        method: Name of the group-contribution method that produced the counts.
    """
    method_group_names = group_names(method)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=["Common_Name", *method_group_names]
        )
        writer.writeheader()
        for compound_name, _smiles, counts in decompositions:
            writer.writerow(
                {
                    "Common_Name": compound_name,
                    **{group: counts.get(group, 0) for group in method_group_names},
                }
            )


def read_decomposition_input(input_path: Path) -> list[tuple[str, str]]:
    """Read reference compound names and SMILES strings from an input CSV.

    The file must have a ``SMILES`` column. ``Reference Compound`` is optional and
    defaults to an empty string. Rows with an empty SMILES value are ignored.

    Raises:
        ValueError: If the file does not contain a ``SMILES`` column.
        OSError: If the file cannot be read.
    """
    with input_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or "SMILES" not in reader.fieldnames:
            raise ValueError(f"{input_path} must contain a SMILES column.")
        return [
            (row.get("Reference Compound") or "", row["SMILES"])
            for row in reader
            if row.get("SMILES")
        ]


__all__ = [
    "UnsupportedGroupError",
    "decompose",
    "decompose_gani",
    "format_decomposition",
    "group_names",
    "read_decomposition_input",
    "write_decompositions_csv",
]
