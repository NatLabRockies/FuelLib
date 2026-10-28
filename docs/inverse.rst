Inverse Fuel Design
===================
The inverse fuel design approach aims to determine the optimal fuel composition that satisfies specific property
constraints using `JAX <https://jax.readthedocs.io/>`_ for automatic differentiation and `Optax <https://optax.readthedocs.io/>`_
for optimization.

Installation and Usage
----------------------
The ``inverse`` module is installable as part of the FuelLib package via pip:

.. code-block:: bash

   pip install "fuellib[inverse]"

Python Usage
^^^^^^^^^^^^
After installation, the ``inverse`` module can be imported and used within your Python
scripts as part of FuelLib:

.. code-block:: python

    from fuellib import Fuel
    from fuellib.simulate.inverse import Problem, d7566_config, solve

    fuel = Fuel(...)
    # Define the inverse design problem with ASTM D7566 specifications
    problem = Problem(config=d7566_config, fuel=fuel)
    solution = solve(problem)
    solution.blend.to_csv("new_gc.csv")  # Optimized blend
    solution.additive.to_csv("additive_gc.csv")  # Additive composition
    print(f"Additive mass fraction: {solution.additive_fraction:.4f}")

Command-Line Usage
^^^^^^^^^^^^^^^^^^
``fl-inverse`` provides a command-line interface for solving inverse fuel design problems using FuelLib. A simple example is:

.. code-block:: bash

   fl-inverse --config d7566 --fuel-name fuel --output fuel_opt

This creates the directory ``fuel_opt`` containing ``fuel_opt.csv`` (the optimized blend),
``additive.csv`` (the additive composition, normalized to 100 %), and ``blend.txt``
(the mass percentages of the original fuel and the additive in the blend).


For more information on usage, run:

.. code-block:: bash

   fl-inverse --help

Generating an Ideal Mixture
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Instead of blending an additive into an original fuel composition to meet specific property constraints, users can
determine an optimal mixture of a fuel's components from scratch that satisfies the specified property constraints.
In this case, the original composition is ignored and the additive mass fraction is automatically set to 1, so the
blend is made entirely of the additive. This is enabled by either:

- passing ``ignore_composition=True`` to :class:`~fuellib.simulate.inverse.models.Problem`
  (or ``--ignore-composition`` to ``fl-inverse``), or
- providing a GCxGC file containing the available reference components with all ``Weight %`` values set to zero.

.. code-block:: python

    problem = Problem(config=d7566_config, fuel=fuel, ignore_composition=True)

.. code-block:: bash

   fl-inverse --fuel-name fuel --ignore-composition

Defining the Problem
--------------------
Constraints
^^^^^^^^^^^
The inverse design problem allows specifying various constraints on fuel properties,
including density, viscosity, and other physical or chemical properties. Each constraint
is represented by a corresponding class, such as :class:`DensityConstraint` or 
:class:`ViscosityConstraint`, and can be customized with specific parameters and
weights:

.. code-block:: python

    from fuellib.simulate.inverse import AromaticsConstraint, DensityConstraint, ViscosityConstraint

    density_constraint = DensityConstraint(
        temp=Units.Quantity(15.0, "celsius"),
        min=Units.Quantity(775.0, "kg/m^3"),
        max=Units.Quantity(840.0, "kg/m^3"),
        weight=1.0,
    )
    viscosity_constraint = ViscosityConstraint(
        temp=Units.Quantity(-20.0, "celsius"),
        max=Units.Quantity(4.0, "mm^2/s"),
        weight=1.0,
    )
    aromatics_constraint = AromaticsConstraint(
        min=Units.Quantity(8.0, "percent"),
        max=Units.Quantity(25.0, "percent"),
        weight=1.0,
    )

Density and viscosity constraints are defined at a temperature and may be given several
times (one per temperature). The aromatics content is evaluated at standard temperature,
so :class:`AromaticsConstraint` takes no ``temp`` and at most one may be given.

In general, constraints contain parameters that define the target value or range for the
property. Constraints are computed using the following correlations and mixing rules:

=========================  ===================================================  ====================================================
Property                   Correlation                                          Mixing Rule
=========================  ===================================================  ====================================================
``DensityConstraint``      N/A                                                  Linear
``ViscosityConstraint``    Dutt\ :footcite:p:`viswanath_viscosity_2007`         Kendall-Monroe\ :footcite:p:`kendall_viscosity_1917`
``AromaticsConstraint``    N/A                                                  Liquid volume fraction of aromatics
=========================  ===================================================  ====================================================

See the :class:`API documentation <fuellib.simulate.inverse.constraints>` for
more details on each constraint class and its parameters.

Config
^^^^^^
The :class:`~fuellib.simulate.inverse.models.Config` object specifies the target properties
and constraints for the fuel design, such as density or viscosity. For more information
on constraints, see the Constraints section above. See the
:class:`API documentation <fuellib.simulate.inverse.models.Config>` for the full list
of configuration options.

Example:

.. code-block:: python

    from fuellib.simulate.inverse import (
        AromaticsConstraint,
        Config,
        DensityConstraint,
        ViscosityConstraint,
    )

    density_specs = (DensityConstraint, ...,)
    viscosity_specs = (ViscosityConstraint, ...,)

    custom_config = Config(
        density=density_specs,
        viscosity=viscosity_specs,
        aromatics=AromaticsConstraint(...),
        ... # Optimization configurations
    )

Problem
^^^^^^^
The :class:`~fuellib.simulate.inverse.models.Problem` object contains the ``Config`` (see the Config section above)
and the initial fuel object to be used in the inverse design process. It serves as the main input for the optimization routine,
providing an interface to the fuel and constraints for evaluating the fuel mixture during the optimization process.

Optimization Process
--------------------
The optimization process involves minimizing target property deviations subject to the constraints defined in the ``Config``.
The solver uses an augmented Lagrangian approach, where the Lagrangian is approximately
minimized for fixed multipliers (inner loop), followed by updates to the multipliers and penalty parameters (outer loop).
The process continues until the maximum number of steps is reached or the constraint violations fall below the specified tolerance.

Blend Parameterization
^^^^^^^^^^^^^^^^^^^^^^
Rather than optimizing the mass fractions directly, the solver blends an *additive* into the
base fuel composition :math:`\mathbf{Y}_0` (:meth:`~fuellib.simulate.inverse.models.Problem.blend`).
The addition is described by two unconstrained parameters: a vector of logits
:math:`\mathbf{z} \in \mathbb{R}^N` (one per component) and a scalar :math:`s` that sets the mass
fraction of the addition in the blend:

.. math::

   \mathbf{Y}_\mathrm{add} = \Pi_\Delta(\mathbf{z}), \qquad
   \alpha = \sigma(s) = \frac{1}{1 + e^{-s}}, \qquad
   \mathbf{Y} = (1 - \alpha)\,\mathbf{Y}_0 + \alpha\,\mathbf{Y}_\mathrm{add},

where :math:`\Pi_\Delta` is the Euclidean projection onto the probability simplex
:math:`\{\mathbf{y} : y_i \ge 0,\ \sum_i y_i = 1\}`. This guarantees that every candidate
:math:`\mathbf{Y}` is a valid composition (all non-negative entries and summing to one) and,
because the simplex projection sets small entries exactly to zero, tends to produce
additions made of only a few components. The parameters are initialized to
:math:`\mathbf{z} = \mathbf{0}` (uniform addition) and :math:`s = -5`
(:math:`\alpha \approx 0.7\%`), so the search starts close to the base fuel.

Constraint Functions
^^^^^^^^^^^^^^^^^^^^
Each active bound of a constraint contributes one normalized constraint function
:math:`g_j(\mathbf{Y})`, which is zero when the bound is satisfied and positive otherwise.
For a mixture property :math:`p(\mathbf{Y})` evaluated at the constraint temperature,

.. math::

   g_j^\mathrm{min} = \frac{\max\left(p_\mathrm{min} - p(\mathbf{Y}),\, 0\right)}{p_\mathrm{ref}},
   \qquad
   g_j^\mathrm{max} = \frac{\max\left(p(\mathbf{Y}) - p_\mathrm{max},\, 0\right)}{p_\mathrm{ref}},

where :math:`p_\mathrm{ref}` is :math:`p_\mathrm{max}` if an upper bound is set and
:math:`p_\mathrm{min}` otherwise. The normalization makes violations of different
properties comparable, so a single tolerance applies to all of them. Each constraint's
``weight`` :math:`w_j` scales its contribution to the penalty.

Augmented Lagrangian
^^^^^^^^^^^^^^^^^^^^
The inner loop minimizes the augmented Lagrangian for
inequality constraints :math:`g_j \le 0`, computed by
:meth:`~fuellib.simulate.inverse.models.Problem.loss`:

.. math::

   \mathcal{L}(\mathbf{z}, s;\, \boldsymbol{\lambda}, \mu) =
   \sum_j \frac{\max\left(\lambda_j + \mu w_j g_j(\mathbf{Y}),\, 0\right)^2 - \lambda_j^2}{2 \mu w_j},

where :math:`\lambda_j \ge 0` are the Lagrange multiplier estimates, :math:`\mu > 0` is
the penalty parameter, and :math:`w_j \gt 0` are the constraint weights. Gradients of
:math:`\mathcal{L}` are computed with JAX automatic differentiation, and the parameters
are updated with `optax`'s Adam optimizer.

Algorithm
^^^^^^^^^
The full procedure implemented in :func:`~fuellib.simulate.inverse.solve` is:

1. Initialize :math:`\mathbf{z} = \mathbf{0}`, :math:`s = -5`,
   :math:`\boldsymbol{\lambda} = \mathbf{0}`, and :math:`\mu` = ``penalty_init``.
2. **Inner loop:** take Adam steps on :math:`\mathcal{L}` with fixed
   :math:`\boldsymbol{\lambda}` and :math:`\mu` until one of the following occurs:

   - ``max_inner_steps`` steps have been taken in this inner loop,
   - the loss has stalled, i.e., its change over the last ``stall_window`` steps is at
     most ``stall_rtol`` times its current magnitude, or
   - the total number of steps reaches ``max_steps``.

3. **Convergence check:** evaluate the constraints and the maximum violation
   :math:`v = \max_j \max(g_j, 0)`. If :math:`v \le` ``violation_tol``, stop.
4. **Multiplier update:**
   :math:`\lambda_j \leftarrow \max\left(\lambda_j + \mu w_j g_j,\, 0\right)`.
5. **Penalty update:** if the violation did not decrease sufficiently, i.e.,
   :math:`v >` ``violation_decrease_ratio`` :math:`\times\, v_\mathrm{prev}`, increase the
   penalty: :math:`\mu \leftarrow \min(` ``penalty_growth`` :math:`\times\, \mu,`
   ``penalty_max`` :math:`)`.
6. If the total number of steps is below ``max_steps``, return to step 2.

The Adam optimizer state is carried across outer iterations. The progress of each outer
iteration (step count, loss, maximum violation, and penalty) is printed to the console.

Solver Settings
^^^^^^^^^^^^^^^
The solver behavior is controlled by the following ``Config`` fields, in addition to the
``max_steps`` argument of :func:`~fuellib.simulate.inverse.solve`
(default 100,000) and the ``log_every`` argument (default 1000), which sets the number
of steps between records of the additive composition and mass fraction histories:

============================  ===========  ===================================================
Setting                       Default      Description
============================  ===========  ===================================================
``learning_rate``             1e-2         Adam learning rate.
``max_inner_steps``           2000         Max Adam steps per outer iteration.
``stall_window``              200          Steps over which the loss change is measured.
``stall_rtol``                1e-6         Relative loss change below which the inner loop
                                           is considered stalled.
``violation_tol``             1e-4         Max normalized violation counted as satisfied.
``penalty_init``              1.0          Initial penalty :math:`\mu`.
``penalty_growth``            10.0         Factor by which :math:`\mu` grows.
``penalty_max``               1e8          Upper limit on :math:`\mu`.
``violation_decrease_ratio``  0.25         Required ratio of the new to the previous max
                                           violation; :math:`\mu` grows if it isn't met.
============================  ===========  ===================================================

Packaged Configs
^^^^^^^^^^^^^^^^
The inverse design routine currently comes with a pre-packaged configuration describing ASTM D7566 fuel 
specifications\ :footcite:`astm_D7566-26`, including:

+---------------+-------------+--------------+--------------+
| Property      | Temperature | Minimum      | Maximum      |
+===============+=============+==============+==============+
| ``Density``   | 15 °C       | 775.0 kg/m³  | 840.0 kg/m³  |
+---------------+-------------+--------------+--------------+
| ``Viscosity`` | -20 °C      | \-           | 8.0 mm²/s    |
|               +-------------+--------------+--------------+
|               | -40 °C      | \-           | 12.0 mm²/s   |
+---------------+-------------+--------------+--------------+


Results
^^^^^^^
On completion, :func:`~fuellib.simulate.inverse.solve` prints any constraints that
remain violated along with the final blend density and viscosity at each constraint
temperature. It returns a :class:`~fuellib.simulate.inverse.models.Solution` containing
the optimized blend (``blend``) and the additive (``additive``) as copies of the fuel's
``gcData`` DataFrame with updated ``Weight %`` columns, the additive mass fraction in the
blend (``additive_fraction``), and histories of the additive composition and mass fraction.
:meth:`~fuellib.simulate.inverse.models.Solution.save` writes the blend, additive, and
blend ratio to files. The optimized blend can be used to create a new
:class:`~fuellib.Fuel` (see :doc:`tutorials-custom-fuels`).

.. footbibliography::