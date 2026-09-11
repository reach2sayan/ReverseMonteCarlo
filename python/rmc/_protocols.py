"""The contracts a Python-defined constraint or move generator meets."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ._core import Constraint, MoveGenerator

__all__ = ["ConstraintProtocol", "MoveGeneratorProtocol", "python_constraint", "python_generator"]


@runtime_checkable
class ConstraintProtocol(Protocol):
    """What rmc.Constraint(obj) needs.

    compute_error(coords, moved) returns the error for coords -- a read-only
    (N, 3) view, valid for the call only -- given the uint64 indices moved
    since the previous call. Optional attributes, read once when wrapped:
    name (default: the class name), cost (evaluation order, cheapest first;
    default 1.0), rigid (a hard gate that adds nothing to chi^2; default
    False), and the methods set_boundary_conditions(bc) and initialise().
    """

    def compute_error(self, coords: NDArray[np.float64], moved: NDArray[np.uint64]) -> float: ...


@runtime_checkable
class MoveGeneratorProtocol(Protocol):
    """What rmc.MoveGenerator(obj) needs.

    generate(coords, indices) moves the atoms at indices by writing into
    coords, a writable (N, 3) view of the frame being moved. Optional:
    modifies_species (read once when wrapped), and rejection_override(),
    returning the generator's own accept/reject verdict or None.
    """

    def generate(self, coords: NDArray[np.float64], indices: NDArray[np.uint64]) -> None: ...


def python_constraint(obj: ConstraintProtocol) -> Constraint:
    """obj as a Constraint an engine can hold."""
    return Constraint(obj)


def python_generator(obj: MoveGeneratorProtocol) -> MoveGenerator:
    """obj as a MoveGenerator a Group can hold."""
    return MoveGenerator(obj)
