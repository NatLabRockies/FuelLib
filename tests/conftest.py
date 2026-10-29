"""Skip collection of the inverse tests when the optional extra is not installed."""

import importlib.util

collect_ignore = []
if any(importlib.util.find_spec(m) is None for m in ("jax", "jaxtyping", "optax")):
    collect_ignore.append("test_inverse.py")
