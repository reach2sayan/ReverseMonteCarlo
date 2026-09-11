"""Views and borrowed objects keep their owner alive; copies are independent."""

from __future__ import annotations

import copy
import gc
from collections.abc import Callable

import numpy as np
import pytest

import rmc


def test_coordinate_view_keeps_structure_alive() -> None:
    view = rmc.AtomicStructure(np.ones((3, 3)), ["Cu"] * 3).coordinates
    gc.collect()
    assert view.base is not None
    np.testing.assert_array_equal(view, np.ones((3, 3)))
    view[0, 0] = 2.0  # still writable into the living structure


def test_atomic_numbers_view_keeps_structure_alive() -> None:
    view = rmc.AtomicStructure(np.zeros((2, 3)), ["O", "H"]).atomic_numbers
    gc.collect()
    assert view.tolist() == [8, 1]


def test_structure_keeps_random_structure_alive() -> None:
    atoms = rmc.make_random_amorphous(["Cu", "Zr"], [8, 8], seed=1).structure
    gc.collect()
    assert len(atoms) == 16
    assert atoms.coordinates.shape == (16, 3)


def test_box_view_keeps_bc_alive() -> None:
    box = rmc.PeriodicBC(4.0 * np.eye(3)).box
    gc.collect()
    np.testing.assert_array_equal(box, 4.0 * np.eye(3))


@pytest.mark.parametrize(
    "duplicate",
    [copy.copy, copy.deepcopy, rmc.AtomicStructure.copy],
    ids=["copy", "deepcopy", "method"],
)
def test_copies_are_independent(
    square: rmc.AtomicStructure,
    duplicate: Callable[[rmc.AtomicStructure], rmc.AtomicStructure],
) -> None:
    twin = duplicate(square)
    twin.coordinates[0, 0] = 42.0
    twin.elements = ["Ag"] * 4
    assert square.coordinates[0, 0] == 0.0
    assert square.elements[0] == "Cu"


def test_list_properties_return_copies(square: rmc.AtomicStructure) -> None:
    elements = square.elements
    elements[0] = "Ag"
    assert square.elements[0] == "Cu"
