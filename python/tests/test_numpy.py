"""AtomicStructure, BoundaryConditions and the enums, as numpy sees them."""

from __future__ import annotations

import enum

import numpy as np
import pytest

import rmc


def test_coordinates_are_a_writable_zero_copy_view(square: rmc.AtomicStructure) -> None:
    view = square.coordinates
    assert view.dtype == np.float64
    assert view.shape == (4, 3)
    assert view.flags.c_contiguous and view.flags.writeable
    view[1, 0] = 3.0
    assert square.coordinates[1, 0] == 3.0
    assert square.distance(0, 1, rmc.InfiniteBC()) == pytest.approx(3.0)


def test_atomic_numbers_default_from_symbols(square: rmc.AtomicStructure) -> None:
    z = square.atomic_numbers
    assert z.dtype == np.int32
    assert z.tolist() == [29, 40, 29, 40]


def test_metadata_defaults_match_the_readers(square: rmc.AtomicStructure) -> None:
    assert square.names == square.elements == square.residues == ["Cu", "Zr", "Cu", "Zr"]
    assert square.molecule_ids == [1, 1, 1, 1]


def test_explicit_metadata() -> None:
    water = rmc.AtomicStructure(
        np.zeros((3, 3)),
        ["O", "H", "H"],
        atomic_numbers=[8, 1, 1],
        names=["OW", "HW1", "HW2"],
        residues=["SOL"] * 3,
        molecule_ids=[7, 7, 7],
    )
    assert water.names == ["OW", "HW1", "HW2"]
    assert water.residues == ["SOL"] * 3
    assert water.molecule_ids == [7, 7, 7]
    assert water.atomic_numbers.tolist() == [8, 1, 1]


def test_unknown_symbol_gets_code_zero() -> None:
    s = rmc.AtomicStructure(np.zeros((1, 3)), ["Xx"])
    assert s.atomic_numbers.tolist() == [0]


def test_metadata_options_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        rmc.AtomicStructure(np.zeros((1, 3)), ["Cu"], [29])  # type: ignore[misc]


def test_coordinate_setter_copies_in_place(square: rmc.AtomicStructure) -> None:
    view = square.coordinates
    new = np.arange(12, dtype=float).reshape(4, 3)
    square.coordinates = new
    np.testing.assert_array_equal(view, new)  # the old view sees it: same buffer


@pytest.mark.parametrize("rows", [3, 5])
def test_coordinate_setter_refuses_a_new_atom_count(
    square: rmc.AtomicStructure, rows: int
) -> None:
    with pytest.raises(ValueError, match="atom count"):
        square.coordinates = np.zeros((rows, 3))


def test_coordinates_need_three_columns() -> None:
    with pytest.raises(TypeError):
        rmc.AtomicStructure(np.zeros((4, 2)), ["Cu"] * 4)


def test_atomic_numbers_setter(square: rmc.AtomicStructure) -> None:
    square.atomic_numbers = np.array([1, 2, 3, 4])
    assert square.atomic_numbers.tolist() == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="atomic_numbers has 2 entries for 4 atoms"):
        square.atomic_numbers = np.array([1, 2])


def test_per_atom_lists_check_length(square: rmc.AtomicStructure) -> None:
    square.elements = ["Ag"] * 4
    assert square.elements == ["Ag"] * 4
    with pytest.raises(ValueError, match="elements has 3 entries for 4 atoms"):
        square.elements = ["Ag"] * 3
    square.molecule_ids = [1, 1, 2, 2]
    assert square.molecule_ids == [1, 1, 2, 2]


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"atomic_numbers": [29]}, "atomic_numbers"),
        ({"names": ["A"]}, "names"),
        ({"residues": ["R"]}, "residues"),
        ({"molecule_ids": [1]}, "molecule_ids"),
    ],
)
def test_constructor_checks_every_per_atom_field(kwargs: dict, field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} has 1 entries for 2 atoms"):
        rmc.AtomicStructure(np.zeros((2, 3)), ["Cu", "Cu"], **kwargs)
    with pytest.raises(ValueError, match="elements has 1 entries for 2 atoms"):
        rmc.AtomicStructure(np.zeros((2, 3)), ["Cu"])


def test_distance_is_bounds_checked(square: rmc.AtomicStructure) -> None:
    with pytest.raises(IndexError):
        square.distance(0, 4, rmc.InfiniteBC())


def test_len_and_repr(square: rmc.AtomicStructure) -> None:
    assert len(square) == 4
    assert repr(square) == 'AtomicStructure(4 atoms, {"Cu": 2, "Zr": 2})'


def test_periodic_wrap_and_min_image(cube_bc: rmc.PeriodicBC) -> None:
    assert isinstance(cube_bc, rmc.BoundaryConditions)
    assert cube_bc.periodic
    assert cube_bc.volume == pytest.approx(1000.0)
    np.testing.assert_allclose(cube_bc.wrap([12.0, -1.0, 5.0]), [2.0, 9.0, 5.0])
    np.testing.assert_allclose(cube_bc.min_image([9.0, -6.0, 0.0]), [-1.0, 4.0, 0.0])
    np.testing.assert_allclose(cube_bc.box @ cube_bc.inv_box, np.eye(3), atol=1e-12)
    assert not cube_bc.box.flags.writeable
    assert repr(cube_bc) == "PeriodicBC(a=10, b=10, c=10, volume=1000)"


def test_periodic_distance_uses_minimum_image(cube_bc: rmc.PeriodicBC) -> None:
    pair = rmc.AtomicStructure(np.array([[0.5, 0, 0], [9.5, 0, 0]]), ["Cu", "Cu"])
    assert pair.distance(0, 1, cube_bc) == pytest.approx(1.0)


def test_infinite_bc_is_the_identity() -> None:
    bc = rmc.InfiniteBC(volume=50.0)
    assert not bc.periodic
    assert bc.volume == 50.0
    np.testing.assert_allclose(bc.wrap([12.0, -1.0, 5.0]), [12.0, -1.0, 5.0])
    assert repr(bc) == "InfiniteBC(volume=50)"
    assert not rmc.BoundaryConditions().periodic
    assert rmc.BoundaryConditions(4.0 * np.eye(3)).periodic


def test_enums_are_int_enums_with_cpp_names() -> None:
    expected = {
        rmc.SymmetryAxis: ["X", "Y", "Z"],
        rmc.MoveGenKind: ["Random", "Langevin", "Leapfrog"],
        rmc.RecursiveMode: ["Refine", "Explore"],
        rmc.StructFormat: ["Pdb", "Vasp", "Lammps"],
        rmc.LammpsAtomStyle: ["Atomic", "Charge", "Molecular", "Full"],
        rmc.DistanceScope: ["Inter", "Intra"],
    }
    for kind, names in expected.items():
        assert issubclass(kind, enum.IntEnum)
        assert [member.name for member in kind] == names
        assert [int(member) for member in kind] == list(range(len(names)))


def test_make_random_amorphous(cuzr: rmc.RandomStructure) -> None:
    atoms = cuzr.structure
    assert len(atoms) == 64
    assert atoms.elements.count("Cu") == atoms.elements.count("Zr") == 32
    side = np.diag(cuzr.box)
    np.testing.assert_allclose(cuzr.box, np.diag(side))
    assert (side > 0).all()
    assert cuzr.periodic_bc().periodic


def test_random_structure_is_seeded() -> None:
    def make() -> rmc.AtomicStructure:
        return rmc.make_random_amorphous(["Cu", "Zr"], [4, 4], seed=3).structure

    a, b = make(), make()
    assert a.elements == b.elements
    np.testing.assert_array_equal(a.coordinates, b.coordinates)


def test_random_structure_options_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        rmc.make_random_amorphous(["Cu"], [1], 3.0)  # type: ignore[misc]


def test_random_structure_reports_bad_input() -> None:
    with pytest.raises(rmc.RandomStructureError, match="must match") as info:
        rmc.make_random_amorphous(["Cu", "Zr"], [4])
    assert isinstance(info.value, rmc.RmcError)


def test_parallel_knobs() -> None:
    assert isinstance(rmc.has_tbb, bool)
    assert rmc.allocated_cpus() >= 1
    assert rmc.default_concurrency() >= 1
    with pytest.raises(ValueError):
        rmc.set_max_concurrency(0)
