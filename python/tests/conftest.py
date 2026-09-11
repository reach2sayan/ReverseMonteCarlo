"""Small structures shared by the suites."""

from __future__ import annotations

import numpy as np
import pytest

import rmc


@pytest.fixture
def square() -> rmc.AtomicStructure:
    """Four atoms on a unit square in the z = 0 plane, alternating Cu/Zr."""
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    return rmc.AtomicStructure(coords, ["Cu", "Zr", "Cu", "Zr"])


@pytest.fixture
def cube_bc() -> rmc.PeriodicBC:
    """A 10 Angstrom cubic cell."""
    return rmc.PeriodicBC(10.0 * np.eye(3))


@pytest.fixture
def cuzr() -> rmc.RandomStructure:
    """64 atoms of Cu50Zr50, seeded."""
    return rmc.make_random_amorphous(["Cu", "Zr"], [32, 32], seed=7)
