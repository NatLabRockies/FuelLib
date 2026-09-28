"""Group Contribution Method (GCM) related functionality for the Fuellib project."""

from .core import GCMRegistry
from .gani import gani_gcm

__all__ = ["GCMRegistry", "gani_gcm"]
