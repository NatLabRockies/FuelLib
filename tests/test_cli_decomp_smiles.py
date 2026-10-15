"""Tests for the fl-decomp-smiles command-line tool."""

import csv
import sys

import pytest

from fuellib.cli.decomp_smiles import main


def test_writes_named_decompositions(monkeypatch, capsys, tmp_path):
    """The CLI should assign one supplied name to each SMILES and export the counts."""
    output_path = tmp_path / "gani.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fl-decomp-smiles",
            "CCc1ccccc1",
            "CCCCCCC",
            "--name",
            "ethylbenzene",
            "n-heptane",
            "--method",
            "gani",
            "--output-file",
            str(output_path),
        ],
    )

    assert main() == 0
    output = capsys.readouterr().out
    assert "ethylbenzene (CCc1ccccc1):" in output
    assert "n-heptane (CCCCCCC):" in output

    with output_path.open(newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert rows[0]["Common_Name"] == "ethylbenzene"
    assert rows[0]["ACH"] == "5"
    assert rows[1]["Common_Name"] == "n-heptane"
    assert rows[1]["CH2"] == "5"


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
        ["CCCCCCC", "CCc1ccccc1", "--name", "one-name"],
        ["CCCCCCC", "--method", "unknown"],
    ],
)
def test_rejects_invalid_arguments(monkeypatch, arguments):
    """The CLI should report invalid sources, names, and methods as usage errors."""
    monkeypatch.setattr(sys, "argv", ["fl-decomp-smiles", *arguments])

    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
