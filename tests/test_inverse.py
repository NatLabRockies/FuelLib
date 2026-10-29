"""Tests for the inverse fuel composition modeling module."""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fuellib.cli.inverse import main
from fuellib import DEFAULT_DATA_DIR, Fuel, Units
from fuellib.simulate.inverse import (
    Config,
    DensityConstraint,
    Problem,
    ViscosityConstraint,
    d7566_config,
    solve,
)

T15 = Units.Quantity(15.0, "celsius")
T_20 = Units.Quantity(-20.0, "celsius")


@pytest.fixture(scope="module")
def fuel() -> Fuel:
    """Multi-component fuel that satisfies the D7566 limits."""
    return Fuel("posf10325")


@pytest.fixture(scope="module")
def Y0(fuel: Fuel) -> jnp.ndarray:
    """Original composition of the fuel."""
    return jnp.asarray(fuel.Y_0.m_as(""))


@pytest.fixture(scope="module")
def problem(fuel: Fuel) -> Problem:
    """Problem using the D7566 limits."""
    return Problem(fuel, d7566_config)


class TestConstraints:
    """Tests for the density and viscosity constraints."""

    @pytest.mark.parametrize("cls", [DensityConstraint, ViscosityConstraint])
    def test_requires_a_bound(self, cls: type) -> None:
        """At least one of min or max must be set."""
        with pytest.raises(ValidationError):
            cls(temp=T15)

    def test_min_must_be_below_max(self) -> None:
        """Min >= max is rejected."""
        with pytest.raises(ValidationError):
            DensityConstraint(
                temp=T15,
                min=Units.Quantity(800.0, "kg/m^3"),
                max=Units.Quantity(700.0, "kg/m^3"),
            )

    def test_labels(self) -> None:
        """One label per active bound, none when the weight is zero."""
        d = DensityConstraint(
            temp=T15,
            min=Units.Quantity(700.0, "kg/m^3"),
            max=Units.Quantity(800.0, "kg/m^3"),
        )
        v = ViscosityConstraint(temp=T_20, max=Units.Quantity(4.0, "mm^2/s"))
        assert len(d.labels()) == 2
        assert len(v.labels()) == 1
        assert d.model_copy(update={"weight": 0.0}).labels() == []

    def test_density_penalty(self, fuel: Fuel, Y0: jnp.ndarray) -> None:
        """Penalty is zero when feasible and positive when violated."""
        ok = DensityConstraint(
            temp=T15,
            min=Units.Quantity(775.0, "kg/m^3"),
            max=Units.Quantity(840.0, "kg/m^3"),
        ).bind(fuel)
        bad = DensityConstraint(temp=T15, min=Units.Quantity(900.0, "kg/m^3")).bind(
            fuel
        )
        assert all(float(g) == 0.0 for g in ok(Y0))
        assert float(bad(Y0)[0]) > 0.0

    def test_viscosity_penalty(self, fuel: Fuel, Y0: jnp.ndarray) -> None:
        """Penalty is zero when feasible and positive when violated."""
        ok = ViscosityConstraint(temp=T_20, max=Units.Quantity(4.0, "mm^2/s")).bind(
            fuel
        )
        bad = ViscosityConstraint(temp=T_20, max=Units.Quantity(1.0, "mm^2/s")).bind(
            fuel
        )
        assert float(ok(Y0)[0]) == 0.0
        assert float(bad(Y0)[0]) > 0.0


class TestConfig:
    """Tests for the configuration model."""

    def test_defaults(self) -> None:
        """Default config has no constraints."""
        c = Config()
        assert c.density == ()
        assert c.viscosity == ()

    def test_d7566_config(self) -> None:
        """D7566 config has one density and two viscosity limits."""
        assert len(d7566_config.density) == 1
        assert len(d7566_config.viscosity) == 2

    def test_rejects_invalid_values(self) -> None:
        """Invalid settings and unknown fields are rejected."""
        with pytest.raises(ValidationError):
            Config(penalty_growth=1.0)
        with pytest.raises(ValidationError):
            Config(not_a_field=1)  # ty: ignore[unknown-argument]


class TestProblem:
    """Tests for the inverse design problem."""

    def test_constraint_names(self, problem: Problem) -> None:
        """Names and weights match the active constraint terms."""
        assert len(problem.constraint_names) == 4
        assert problem.constraint_weights.shape == (4,)

    def test_blend(self, problem: Problem) -> None:
        """Blend is a valid composition that reduces to Y_0 without addition."""
        params = {
            "logits": jnp.zeros_like(problem.Y_0),
            "frac": jnp.asarray(-50.0),
        }
        Yi = problem.blend(params)
        assert np.isclose(float(Yi.sum()), 1.0)
        assert np.all(Yi >= 0.0)
        assert np.allclose(Yi, problem.Y_0)

    def test_blend_empty_base(self, tmp_path: Path) -> None:
        """With an empty base fuel the blend is the addition and sums to 1."""
        (tmp_path / "gcData").mkdir()
        gc = pd.read_csv(DEFAULT_DATA_DIR / "gcData" / "posf10325.csv")
        gc.assign(**{"Weight %": 0.0}).to_csv(
            tmp_path / "gcData" / "blank.csv", index=False
        )
        problem = Problem(Fuel("blank", fuelDataDir=tmp_path), d7566_config)
        params = {
            "logits": jnp.zeros_like(problem.Y_0),
            "frac": jnp.asarray(-5.0),
        }
        Yi = problem.blend(params)
        assert np.isclose(float(Yi.sum()), 1.0)
        assert np.all(Yi >= 0.0)

    def test_blend_ignore_composition(self, fuel: Fuel) -> None:
        """Ignoring the composition makes the blend the addition itself."""
        problem = Problem(fuel, d7566_config, ignore_composition=True)
        assert np.array_equal(problem.Y_0, np.zeros(fuel.num_compounds))
        params = {
            "logits": jnp.zeros_like(problem.Y_0),
            "frac": jnp.asarray(-5.0),
        }
        Yi = problem.blend(params)
        assert np.allclose(Yi, 1.0 / fuel.num_compounds)

    def test_constraints_and_loss(self, problem: Problem) -> None:
        """Base fuel is feasible, so constraints and loss are zero."""
        params = {
            "logits": jnp.zeros_like(problem.Y_0),
            "frac": jnp.asarray(-50.0),
        }
        g = problem.constraints(problem.blend(params))
        assert g.shape == (len(problem.constraint_names),)
        assert np.all(g <= 0.0)

        lam = jnp.zeros(len(problem.constraint_names))
        loss = problem.loss(params, lam, jnp.asarray(1.0))
        assert loss.shape == ()
        assert float(loss) == 0.0


class TestSolve:
    """Tests for the augmented Lagrangian solver."""

    def test_feasible_fuel(self, fuel: Fuel, problem: Problem) -> None:
        """A fuel that already meets the limits stays close to its composition."""
        result = solve(problem, max_steps=1000).blend
        assert isinstance(result, pd.DataFrame)
        Y = result["Weight %"].to_numpy(dtype=float) / 100.0
        assert np.isclose(Y.sum(), 1.0)
        assert np.allclose(Y, fuel.Y_0.m_as(""), atol=1e-2)

    def test_meets_density_limit(self, fuel: Fuel) -> None:
        """Solver raises the density above a minimum the base fuel misses."""
        rho_min = Units.Quantity(810.0, "kg/m^3")
        config = Config(density=(DensityConstraint(temp=T15, min=rho_min),))
        result = solve(Problem(fuel, config)).blend
        Y = Units.Quantity(result["Weight %"].to_numpy(dtype=float) / 100.0, "")
        assert np.isclose(Y.magnitude.sum(), 1.0)
        rho = fuel.mixture_density(Y, T15).m_as("kg/m^3")
        assert rho >= rho_min.m_as("kg/m^3") * (1 - config.violation_tol)

    def test_without_gc_bin_and_zero_weights(self, tmp_path: Path) -> None:
        """Solver runs on GCxGC data without 'GC-Bin' and with zero 'Weight %'."""
        (tmp_path / "gcData").mkdir()
        gc = pd.read_csv(DEFAULT_DATA_DIR / "gcData" / "posf10325.csv")
        gc.drop(columns="GC-Bin").assign(**{"Weight %": 0.0}).to_csv(
            tmp_path / "gcData" / "components.csv", index=False
        )
        fuel = Fuel("components", fuelDataDir=tmp_path)
        solution = solve(Problem(fuel, d7566_config), max_steps=10)
        assert np.isclose(solution.blend["Weight %"].sum(), 100.0)
        assert solution.additive_fraction == 1.0

    def test_ignore_composition(self, fuel: Fuel) -> None:
        """Ignoring the composition gives an additive fraction of 1."""
        problem = Problem(fuel, d7566_config, ignore_composition=True)
        solution = solve(problem, max_steps=10)
        assert solution.additive_fraction == 1.0
        assert np.all(solution.additive_fraction_history == 1.0)
        assert np.allclose(solution.blend["Weight %"], solution.additive["Weight %"])


class TestCLI:
    """Tests for the fl-inverse command-line tool."""

    def test_writes_outputs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The CLI solves the problem and exports the blend, additive, and ratio."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "fl-inverse",
                "--fuel-name",
                "posf10325",
                "--max-steps",
                "10",
                "--disable-ref-props",
                "--output",
                str(tmp_path),
            ],
        )
        main()
        assert (tmp_path / "posf10325_opt.csv").exists()
        assert (tmp_path / "additive.csv").exists()
        assert "Additive:" in (tmp_path / "blend.txt").read_text()
