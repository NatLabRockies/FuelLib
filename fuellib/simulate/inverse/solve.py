"""Augmented Lagrangian solver for the inverse fuel design problem."""

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from jaxtyping import PyTree
from optax.projections import projection_simplex

import fuellib

from ...utils import types
from ...utils.types import Units
from .constraints import DensityConstraint, ViscosityConstraint
from .models import Problem, Solution

_v = f"v{fuellib.__version__}"
logo = rf"""
 /$$$$$$$$                  /$$ /$$       /$$ /$$                                     
| $$_____/                 | $$| $$      |__/| $$                                     
| $$    /$$   /$$  /$$$$$$ | $$| $$       /$$| $$$$$$$                                
| $$$$$| $$  | $$ /$$__  $$| $$| $$      | $$| $$__  $$                               
| $$__/| $$  | $$| $$$$$$$$| $$| $$      | $$| $$  \ $$                               
| $$   | $$  | $$| $$_____/| $$| $$      | $$| $$  | $$                               
| $$   |  $$$$$$/|  $$$$$$$| $$| $$$$$$$$| $$| $$$$$$$/                               
|__/    \______/  \_______/|__/|________/|__/|_______/       
                                                                                    
         /$$$$$$                                                            
        |_  $$_/                                                            
          | $$   /$$$$$$$  /$$    /$$ /$$$$$$   /$$$$$$   /$$$$$$$  /$$$$$$ 
          | $$  | $$__  $$|  $$  /$$//$$__  $$ /$$__  $$ /$$_____/ /$$__  $$
          | $$  | $$  \ $$ \  $$/$$/| $$$$$$$$| $$  \__/|  $$$$$$ | $$$$$$$$
          | $$  | $$  | $$  \  $$$/ | $$_____/| $$       \____  $$| $$_____/
         /$$$$$$| $$  | $$   \  $/  |  $$$$$$$| $$       /$$$$$$$/|  $$$$$$$
        |______/|__/  |__/    \_/    \_______/|__/      |_______/  \_______/ ({_v})  
"""


def solve(
    problem: Problem, max_steps: int = 100_000, log_every: int = 1000
) -> Solution:
    """Minimize the sparsity objective subject to the property constraints.

    Approximately minimizes the Lagrangian for fixed multipliers (inner loop),
    then updates the multipliers and, if the violation didn't drop enough,
    increases the penalty (outer loop).

    Args:
        problem: Inverse design problem.
        max_steps: Max optimization steps.
            Default is 100,000.
        log_every: Number of steps between records of the additive composition
            and mass fraction histories. The final state is always recorded.
            Default is 1,000.

    Returns:
        Solution containing the optimized blend, the additive composition, and
        the mass fraction of the additive in the blend.

    Raises:
        ValueError: If `log_every` is less than 1.
    """
    if log_every < 1:
        raise ValueError(f"log_every must be >= 1, got {log_every}.")
    c = problem.config
    optimizer = optax.adam(learning_rate=c.learning_rate)

    @jax.jit
    def update(
        params: PyTree, opt_state: optax.OptState, lam: Array, mu: Array
    ) -> tuple[PyTree, optax.OptState, Array]:
        loss, grads = jax.value_and_grad(problem.loss)(params, lam, mu)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def evaluate_constraints(params: PyTree) -> Array:
        return problem.constraints(problem.blend(params))

    def violated_names(g: Array) -> list[str]:
        names = [
            name
            for name, v in zip(problem.constraint_names, g, strict=True)
            if v > c.violation_tol
        ]
        return names or ["None"]

    params = {
        "logits": jnp.zeros_like(problem.Y_0),  # Uniform addition composition
        "frac": jnp.asarray(-5.0),  # sigmoid(-5) ~ 0.7% addition: start near Y_0
    }

    opt_state = optimizer.init(params)
    lam = jnp.zeros(len(problem.constraint_names))
    mu = c.penalty_init

    step = 0
    outer = 0
    converged = False
    t_start = time.perf_counter()

    prev_violation = jnp.inf
    logits_hist: list[Array] = []
    frac_hist: list[float] = []
    loss_hist: list[float] = []

    print(logo)
    print("\nOptimization\n" + "=" * 80)
    print(
        f"{'Iter':<6}{'Step':<9}{'Loss':<14}{'Max violation':<16}{'Penalty':<10}"
        f"{'Violated'}\n" + "-" * 80
    )
    while step < max_steps:
        start = step
        while step < max_steps and step - start < c.max_inner_steps:
            new_params, opt_state, loss = update(params, opt_state, lam, mu)

            if step % log_every == 0:
                logits_hist.append(params["logits"])
                frac_hist.append(float(params["frac"]))
            loss_hist.append(float(loss))
            params = new_params
            step += 1

            if step - start > c.stall_window and abs(
                loss_hist[-1 - c.stall_window] - loss_hist[-1]
            ) <= c.stall_rtol * max(abs(loss_hist[-1]), 1e-12):
                break

        outer += 1
        g = evaluate_constraints(params)
        max_violation = float(jnp.maximum(g, 0.0).max(initial=0.0))
        first, *rest = violated_names(g)
        print(
            f"{outer:<6}{step:<9}{loss_hist[-1]:<14.6g}{max_violation:<16.3g}{mu:<10.3g}"
            f"{first}"
        )
        for name in rest:
            print(f"{'':<55}{name}")
        if max_violation <= c.violation_tol:
            converged = True
            break

        lam = jax.nn.relu(lam + mu * problem.constraint_weights * g)
        if max_violation > c.violation_decrease_ratio * prev_violation:
            mu = min(mu * c.penalty_growth, c.penalty_max)
        prev_violation = max_violation
    print("=" * 80)
    elapsed = time.perf_counter() - t_start

    g = evaluate_constraints(params)
    fuel = problem.fuel
    Yi = Units.Quantity(np.array(problem.blend(params)), "")
    Y0 = None if problem._empty_base else Units.Quantity(np.array(problem.Y_0), "")

    def prop_row(
        name: str,
        con: DensityConstraint | ViscosityConstraint,
        prop: Callable[
            [types.Quantity1D, types.Quantity0D], types.Quantity0D | types.Quantity1D
        ],
        units: str,
    ) -> list[str]:
        def fmt_value(v: types.Quantity0D | types.Quantity1D | None) -> str:
            return "-" if v is None else f"{v.m_as(units):.2f}"

        return [
            name,
            f"{con.temp:.1f~P}",
            fmt_value(None if Y0 is None else prop(Y0, con.temp)),
            fmt_value(prop(Yi, con.temp)),
            fmt_value(con.min),
            fmt_value(con.max),
            f"{Units.Unit(units):~P}",
        ]

    headers = ["Property", "Temp", "Original", "Final", "Min", "Max", "Units"]
    rows = [
        prop_row("Density", d, fuel.mixture_density, "kg/m^3") for d in c.density
    ] + [
        prop_row("Viscosity", v, fuel.mixture_kinematic_viscosity, "mm^2/s")
        for v in c.viscosity
    ]
    widths = [max(len(r[i]) for r in [headers, *rows]) for i in range(len(headers))]
    sep = "    "
    width = sum(widths) + len(sep) * (len(widths) - 1)

    def fmt(row: list[str]) -> str:
        # Left-align text columns (Property, Units), right-align numeric columns
        return sep.join(
            f"{cell:<{w}}" if i in (0, len(row) - 1) else f"{cell:>{w}}"
            for i, (cell, w) in enumerate(zip(row, widths, strict=True))
        )

    new_gc = fuel.gcData.copy()
    new_gc["Weight %"] = Yi.magnitude * 100.0

    original = fuel.Y_0.magnitude * 100.0
    final = new_gc["Weight %"].to_numpy(float)
    diff = final - original

    labels = (
        new_gc["GC-Bin"].astype(str).to_list()
        if "GC-Bin" in new_gc.columns
        else list(fuel.compounds)
    )
    gc_headers = ["Compound", "Original %", "Final %", "Diff %"]
    gc_rows = [
        [name, f"{o:.2f}", f"{f:.2f}", f"{d:+.2f}"]
        for name, o, f, d in zip(labels, original, final, diff, strict=True)
    ]
    gc_total = ["Total", f"{original.sum():.2f}", f"{final.sum():.2f}", ""]

    gc_widths = [
        max(len(r[i]) for r in [gc_headers, *gc_rows, gc_total])
        for i in range(len(gc_headers))
    ]
    gc_width = sum(gc_widths) + len(sep) * (len(gc_widths) - 1)

    def gc_fmt(row: list[str]) -> str:
        cells = [f"{row[0]:<{gc_widths[0]}}"]
        cells += [f"{c:>{w}}" for c, w in zip(row[1:], gc_widths[1:], strict=True)]
        return sep.join(cells)

    print(f"\nFinal GCxGC Data: {fuel.name}")
    print("=" * gc_width)
    print(gc_fmt(gc_headers))
    print("-" * gc_width)
    for row in gc_rows:
        print(gc_fmt(row))
    print("-" * gc_width)
    print(gc_fmt(gc_total))
    print("=" * gc_width)

    print("\nFinal Properties")
    print("=" * width)
    print(fmt(headers))
    print("-" * width)
    for row in rows:
        print(fmt(row))
    print("=" * width)

    Y_add = np.array(projection_simplex(params["logits"]))
    add_gc = fuel.gcData.copy()
    add_gc["Weight %"] = Y_add / Y_add.sum() * 100.0

    alpha = 1.0 if problem._empty_base else float(jax.nn.sigmoid(params["frac"]))

    max_violation = float(jnp.maximum(g, 0.0).max(initial=0.0))
    summary = [
        ("Status", "Converged" if converged else "Not converged (max steps reached)"),
        ("Steps", f"{step:,} / {max_steps:,}"),
        ("Outer iterations", f"{outer}"),
        ("Elapsed time", f"{elapsed:.2f} s"),
        ("Final loss", f"{loss_hist[-1]:.6g}" if loss_hist else "-"),
        ("Max violation", f"{max_violation:.3g} (tol {c.violation_tol:.3g})"),
        ("Violated constraints", ", ".join(violated_names(g))),
        ("Final penalty", f"{mu:.3g}"),
        ("Additive mass fraction", f"{alpha * 100.0:.2f} %"),
        (
            "Additive components",
            f"{np.count_nonzero(Y_add > 1e-8)} / {Y_add.size}",
        ),
    ]
    label_w = max(len(label) for label, _ in summary)
    summary_width = label_w + len(sep) + max(len(value) for _, value in summary)

    print("\nSummary")
    print("=" * summary_width)
    for label, value in summary:
        print(f"{label:<{label_w}}{sep}{value}")
    print("=" * summary_width)

    # Include the final parameters so the history ends at the returned solution
    logits_hist.append(params["logits"])
    frac_hist.append(float(params["frac"]))
    additive_history = np.array(jax.vmap(projection_simplex)(jnp.stack(logits_hist)))
    additive_fraction_history = (
        np.ones(len(frac_hist))
        if problem._empty_base
        else np.array(jax.nn.sigmoid(jnp.asarray(frac_hist)))
    )

    return Solution(
        blend=new_gc,
        additive=add_gc,
        additive_fraction=alpha,
        additive_history=additive_history,
        additive_fraction_history=additive_fraction_history,
    )
