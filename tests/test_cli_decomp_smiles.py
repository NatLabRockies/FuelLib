"""Tests for the fl-decomp-smiles command-line tool."""

import csv
import sys

import pytest

from fuellib.cli.decomp_smiles import decompose_smiles


def test_writes_named_decompositions(monkeypatch, capsys, tmp_path):
    """The CLI should assign one supplied name to each SMILES and export the counts."""
    output_path = tmp_path / "gani.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fl-decomp-smiles",
            "--smiles",
            "CCc1ccccc1",
            "CCCCCCC",
            "--name",
            "ethylbenzene",
            "n-heptane",
            "--method",
            "gani",
            "--out",
            str(output_path),
        ],
    )

    decompose_smiles()
    assert f"written to {output_path}" in capsys.readouterr().out

    with output_path.open(newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert rows[0]["Reference Compound"] == "ethylbenzene"
    assert rows[0]["ACH"] == "5"
    assert rows[1]["Reference Compound"] == "n-heptane"
    assert rows[1]["CH2"] == "5"


def test_reads_input_file(monkeypatch, capsys, tmp_path):
    """The CLI should process all compounds in an input CSV."""
    input_path = tmp_path / "compounds.csv"
    input_path.write_text("Reference Compound,SMILES\nn-heptane,CCCCCCC\n")
    monkeypatch.setattr(sys, "argv", ["fl-decomp-smiles", "-i", str(input_path)])

    decompose_smiles()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["CH3", "CH2"]
    assert lines[-1].split() == ["n-heptane", "2", "5"]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["-s", "CCCCCCC", "CCc1ccccc1", "--name", "one-name"],
        ["-s", "CCCCCCC", "--method", "unknown"],
    ],
)
def test_rejects_invalid_arguments(monkeypatch, arguments):
    """The CLI should report invalid sources, names, and methods as usage errors."""
    monkeypatch.setattr(sys, "argv", ["fl-decomp-smiles", *arguments])

    with pytest.raises(SystemExit) as error:
        decompose_smiles()
    assert error.value.code == 2
