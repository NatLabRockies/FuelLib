"""Tests for the fl-decomp-smiles command-line tool."""

import csv
import sys

import pytest

from fuellib.cli.decomp_smiles import main


def test_writes_named_decomposition(monkeypatch, capsys, tmp_path):
    """The CLI should decompose a named SMILES and export its group counts."""
    output_path = tmp_path / "gani.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fl-decomp-smiles",
            "CCc1ccccc1",
            "--name",
            "ethylbenzene",
            "--method",
            "gani",
            "--output-file",
            str(output_path),
        ],
    )

    assert main() == 0
    assert "ethylbenzene (CCc1ccccc1):" in capsys.readouterr().out

    with output_path.open(newline="") as output_file:
        row = next(csv.DictReader(output_file))
    assert row["Common_Name"] == "ethylbenzene"
    assert row["ACH"] == "5"


def test_reads_input_file(monkeypatch, capsys, tmp_path):
    """The CLI should process all compounds in an input CSV."""
    input_path = tmp_path / "compounds.csv"
    input_path.write_text("Reference Compound,SMILES\nn-heptane,CCCCCCC\n")
    monkeypatch.setattr(sys, "argv", ["fl-decomp-smiles", "-i", str(input_path)])

    assert main() == 0
    assert "n-heptane (CCCCCCC): (2x) CH3, (5x) CH2" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["CCCCCCC", "CCc1ccccc1", "--name", "two-names"],
        ["CCCCCCC", "--method", "unknown"],
    ],
)
def test_rejects_invalid_arguments(monkeypatch, arguments):
    """The CLI should report invalid sources, names, and methods as usage errors."""
    monkeypatch.setattr(sys, "argv", ["fl-decomp-smiles", *arguments])

    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
