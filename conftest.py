"""Skip collection of the optional inverse module when its extra is not installed."""

import importlib.util

collect_ignore = []
if any(importlib.util.find_spec(m) is None for m in ("jax", "jaxtyping", "optax")):
    collect_ignore += [
        "fuellib/simulate/inverse",
        "fuellib/cli/inverse.py",
        "tests/test_inverse.py",
    ]
