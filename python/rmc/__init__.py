"""Reverse Monte Carlo structural refinement.

Refines an atomic structure against experimental G(r), S(Q) and bond-angle
distribution data. The C++ engine lives in :mod:`rmc._core`; import from
:mod:`rmc`, which re-exports it with a stable ``__all__``.
"""

from __future__ import annotations

from . import _core

__version__: str = _core.__version__

__all__ = sorted(["__version__"])
