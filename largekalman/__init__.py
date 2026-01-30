"""Kalman filtering and smoothing for larger-than-memory datasets."""

from .filter import smooth
from .em import em, em_step

__version__ = "0.2.10"
__all__ = ["smooth", "em", "em_step"]
