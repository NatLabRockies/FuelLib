"""Tests for the fl decomp command-line tool."""

import csv
import sys

import pytest

from fuellib.cli import main


def decompose_smiles() -> None:
    """Run the CLI with the patched ``sys.argv``."""
    main()


def test_writes_named_decompositions(monkeypatch, capsys, tmp_path):
    """The CLI should assign one supplied name to each SMILES and export the counts."""
    output_path = tmp_path / "gani.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fl",
            "decomp",
            "--smiles",
            "CCc1ccccc1",
            "CC(C)CCCCC",
            "--name",
            "ethylbenzene",
            "2-methylheptane",
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
    assert rows[1]["Reference Compound"] == "2-methylheptane"
    assert rows[1]["CH2"] == "4"


def test_reads_input_file(monkeypatch, capsys, tmp_path):
    """The CLI should process all compounds in an input CSV."""
    input_path = tmp_path / "compounds.csv"
    input_path.write_text("Reference Compound,SMILES\nn-heptane,CCCCCCC\n")
    monkeypatch.setattr(sys, "argv", ["fl", "decomp", "-i", str(input_path)])

    decompose_smiles()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["CH3", "CH2"]
    assert lines[-1].split() == ["n-heptane", "2", "5"]


def test_nannoolal_method(monkeypatch, capsys):
    """The CLI should decompose with the Nannoolal method when requested."""
    monkeypatch.setattr(
        sys, "argv", ["fl", "decomp", "-s", "CCCCCC", "-m", "nannoolal"]
    )

    decompose_smiles()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["CH3-(ne)", "-C(c)H2-"]
    assert lines[-1].split() == ["CCCCCC", "2", "4"]


@pytest.mark.parametrize("source", ["arguments", "csv"])
def test_names_failing_molecule_in_multi_input_error(
    monkeypatch, capsys, tmp_path, source
):
    """A failed row identifies its name, SMILES, method, and cause without a traceback."""
    if source == "arguments":
        arguments = [
            "fl",
            "decomp",
            "--smiles",
            "CCCCCCC",
            "CCCCCCCO",
            "--name",
            "n-heptane",
            "1-heptanol",
            "--method",
            "gani",
        ]
    else:
        input_path = tmp_path / "compounds.csv"
        input_path.write_text(
            "Reference Compound,SMILES\nn-heptane,CCCCCCC\n1-heptanol,CCCCCCCO\n"
        )
        arguments = [
            "fl",
            "decomp",
            "--input",
            str(input_path),
            "--method",
            "gani",
        ]

    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as error:
        decompose_smiles()

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        "Error decomposing '1-heptanol': "
        "Gani decomposition failed for SMILES 'CCCCCCCO': "
        "first-order groups matched 7 of 8 atoms."
    )


@pytest.mark.parametrize(
    ("method", "prefix"),
    [("gani", "Gani"), ("nannoolal", "Nannoolal"), ("unifac", "UNIFAC")],
)
def test_unnamed_failure_does_not_repeat_the_smiles(
    monkeypatch, capsys, method, prefix
):
    """Without --name the label equals the SMILES, so no label prefix is added."""
    monkeypatch.setattr(sys, "argv", ["fl", "decomp", "-s", "CCCCCCCO", "-m", method])

    with pytest.raises(SystemExit) as error:
        decompose_smiles()

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        f"Error: {prefix} decomposition failed for SMILES 'CCCCCCCO': "
        "first-order groups matched 7 of 8 atoms."
    )


def test_rejects_missing_csv_smiles_with_molecule_context(
    monkeypatch, capsys, tmp_path
):
    """A blank CSV SMILES cell reports its row name without exposing a traceback."""
    input_path = tmp_path / "compounds.csv"
    input_path.write_text(
        "Reference Compound,SMILES\nn-heptane,CCCCCCC\nmissing compound,\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["fl", "decomp", "--input", str(input_path), "--method", "gani"],
    )

    with pytest.raises(SystemExit) as error:
        decompose_smiles()

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        "Error decomposing 'missing compound': "
        "Gani decomposition failed: SMILES must be a non-empty string."
    )


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
    monkeypatch.setattr(sys, "argv", ["fl", "decomp", *arguments])

    with pytest.raises(SystemExit) as error:
        decompose_smiles()
    assert error.value.code == 2


def test_decomposes_inchi_input(monkeypatch, capsys):
    """--from inchi should decompose InChI strings like their SMILES equivalents."""
    inchi = "InChI=1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3"
    monkeypatch.setattr(
        sys,
        "argv",
        ["fl", "decomp", "--from", "inchi", "-s", inchi, "-n", "n-heptane"],
    )

    decompose_smiles()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["CH3", "CH2"]
    assert lines[-1].split() == ["n-heptane", "2", "5"]


def test_reads_inchi_column(monkeypatch, capsys, tmp_path):
    """--from inchi should read the structures from an 'InChI' CSV column."""
    input_path = tmp_path / "compounds.csv"
    input_path.write_text(
        "Reference Compound,InChI\n"
        'n-heptane,"InChI=1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3"\n'
    )
    monkeypatch.setattr(
        sys, "argv", ["fl", "decomp", "--from", "inchi", "-i", str(input_path)]
    )

    decompose_smiles()
    assert capsys.readouterr().out.splitlines()[-1].split() == ["n-heptane", "2", "5"]


def test_rejects_invalid_inchi(monkeypatch, capsys):
    """An unparsable InChI should be reported with its label and exit with code 1."""
    monkeypatch.setattr(
        sys, "argv", ["fl", "decomp", "--from", "inchi", "-s", "not-an-inchi"]
    )

    with pytest.raises(SystemExit) as error:
        decompose_smiles()
    assert error.value.code == 1
    assert "Could not parse InChI 'not-an-inchi'" in capsys.readouterr().err
