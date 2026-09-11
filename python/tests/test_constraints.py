"""Constraints, mirroring tests/test_constraints.cpp, plus the Python-side
pieces: the Constraint holder, Python-defined constraints, ConstraintCollection."""

from __future__ import annotations

import gc
import math

import numpy as np
import pytest

import rmc

ALL2 = [0, 1]


def pair(ax: float, bx: float) -> np.ndarray:
    return np.array([[ax, 0.0, 0.0], [bx, 0.0, 0.0]])


def chain(n: int = 8, spacing: float = 3.0) -> np.ndarray:
    coords = np.zeros((n, 3))
    coords[:, 0] = spacing * np.arange(n)
    return coords


def flat_target(value: float, n_bins: int = 50, dr: float = 0.1) -> np.ndarray:
    r = dr * np.arange(1, n_bins + 1)
    return np.column_stack([r, np.full_like(r, value)])


def pair_function(kind: type, value: float = 0.0, elements: bool = False):
    c = kind()
    c.set_experimental_data(flat_target(value))
    c.set_number_density(0.03)
    if elements:
        c.set_elements(rmc.AtomicStructure(chain(), ["C"] * 8))
    c.initialise()
    return c


# ---- geometric ----


def test_satisfied_bond_has_zero_error() -> None:
    bond = rmc.BondConstraint()
    bond.add_bond(0, 1, 1.0, 2.0)
    assert bond.compute_error(pair(0.0, 1.5), ALL2) == pytest.approx(0.0, abs=1e-8)


def test_too_short_bond_accumulates_error() -> None:
    bond = rmc.BondConstraint()
    bond.add_bond(0, 1, 1.5, 2.5)
    assert bond.compute_error(pair(0.0, 1.0), ALL2) == pytest.approx(0.5, abs=1e-8)


def test_bond_should_reject_after_worsening_move() -> None:
    bond = rmc.BondConstraint()
    bond.add_bond(0, 1, 1.0, 2.0)
    held = rmc.Constraint(bond)
    held.compute_before_move(pair(0.0, 1.5), ALL2)
    held.compute_after_move(pair(0.0, 0.5), ALL2)
    assert held.should_reject()


def test_bond_accept_then_error_before_updates() -> None:
    bond = rmc.BondConstraint()
    bond.add_bond(0, 1, 1.0, 2.0)
    held = rmc.Constraint(bond)
    held.compute_before_move(pair(0.0, 1.2), ALL2)
    held.compute_after_move(pair(0.0, 1.8), ALL2)
    held.accept()
    assert held.standard_error() == pytest.approx(0.0, abs=1e-8)


def test_right_angle_satisfied_and_violated() -> None:
    corner = np.array([[1.0, 0, 0], [0, 0, 0], [0, 1.0, 0]])
    ok = rmc.AngleConstraint()
    ok.add_angle(0, 1, 2, 0.0, math.pi)
    assert ok.compute_error(corner, [0, 1, 2]) == pytest.approx(0.0, abs=1e-8)
    tight = rmc.AngleConstraint()
    tight.add_angle(0, 1, 2, 2.0, math.pi)
    assert tight.compute_error(corner, [0, 1, 2]) > 0.0


def test_zero_dihedral_accepted() -> None:
    line = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    dihedral = rmc.DihedralAngleConstraint()
    dihedral.add_dihedral(0, 1, 2, 3, -math.pi, math.pi)
    assert dihedral.compute_error(line, [0, 1, 2, 3]) == pytest.approx(0.0, abs=1e-8)


def test_flat_improper_accepted() -> None:
    plane = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    improper = rmc.ImproperAngleConstraint()
    improper.add_improper(0, 1, 2, 3, -0.1, 0.1)
    assert improper.compute_error(plane, [0, 1, 2, 3]) == pytest.approx(0.0, abs=1e-8)


def test_flags_and_names() -> None:
    bond = rmc.BondConstraint()
    assert bond.name == "BondConstraint"
    assert bond.is_rigid and not bond.is_singular
    pdf = rmc.PairDistributionConstraint()
    assert pdf.is_singular and not pdf.is_rigid
    assert pdf.computation_cost > bond.computation_cost
    reduced = rmc.ReducedStructureFactorConstraint()
    assert reduced.is_singular and not reduced.is_rigid
    assert reduced.name != rmc.StructureFactorConstraint().name


def test_compute_error_checks_indices() -> None:
    bond = rmc.BondConstraint()
    with pytest.raises(IndexError):
        bond.compute_error(pair(0.0, 1.0), [0, 2])


# ---- distance and coordination ----


@pytest.mark.parametrize(
    ("kind", "molecules", "violated"),
    [
        (rmc.InterMolecularDistanceConstraint, [1, 2], True),
        (rmc.InterMolecularDistanceConstraint, [1, 1], False),
        (rmc.IntraMolecularDistanceConstraint, [1, 1], True),
        (rmc.IntraMolecularDistanceConstraint, [1, 2], False),
    ],
)
def test_minimum_distance_respects_scope(kind: type, molecules: list[int], violated: bool) -> None:
    atoms = rmc.AtomicStructure(pair(0.0, 1.0), ["Cu", "Cu"], molecule_ids=molecules)
    close = kind()
    close.set_minimum_distance("Cu", "Cu", 2.0)
    close.set_structure(atoms)
    assert (close.compute_error(atoms.coordinates, ALL2) > 0.0) == violated
    assert close.is_rigid


def test_borrowed_structure_is_kept_alive() -> None:
    close = rmc.InterMolecularDistanceConstraint()
    close.set_minimum_distance("Cu", "Cu", 2.0)
    close.set_structure(rmc.AtomicStructure(pair(0.0, 1.0), ["Cu", "Cu"], molecule_ids=[1, 2]))
    gc.collect()
    assert close.compute_error(pair(0.0, 1.0), ALL2) > 0.0


def test_coordination_shell(square: rmc.AtomicStructure) -> None:
    def shell(min_cn: int) -> rmc.CoordinationConstraint:
        c = rmc.CoordinationConstraint()
        c.add_shell(0, "Cu", 0.0, 1.5, min_cn, 4)
        c.set_elements(square)
        c.set_boundary_conditions(rmc.InfiniteBC())
        return c

    # Atom 0 (Cu) has one Cu neighbour within 1.5 A: atom 2, at sqrt(2).
    assert shell(1).compute_error(square.coordinates, [0, 1, 2, 3]) == 0.0
    assert shell(2).compute_error(square.coordinates, [0, 1, 2, 3]) > 0.0


# ---- pair functions ----


@pytest.mark.parametrize("kind", [rmc.PairDistributionConstraint, rmc.PairCorrelationConstraint])
def test_pair_error_is_finite_and_non_negative(kind: type) -> None:
    err = pair_function(kind).compute_error(chain(), list(range(8)))
    assert math.isfinite(err) and err >= 0.0


def test_pdf_and_pcf_differ() -> None:
    all8 = list(range(8))
    pdf = pair_function(rmc.PairDistributionConstraint, 1.0).compute_error(chain(), all8)
    pcf = pair_function(rmc.PairCorrelationConstraint, 1.0).compute_error(chain(), all8)
    assert pdf != pcf


def test_curves_have_the_target_grid() -> None:
    pdf = pair_function(rmc.PairDistributionConstraint, 1.0)
    pdf.compute_error(chain(), list(range(8)))
    assert pdf.computed.shape == pdf.experimental.shape == (50,)
    np.testing.assert_allclose(pdf.experimental, 1.0)


def test_spherical_shape_damps_computed_curve() -> None:
    all8 = list(range(8))
    plain = pair_function(rmc.PairDistributionConstraint, elements=True)
    plain.compute_error(chain(), all8)
    shaped = pair_function(rmc.PairDistributionConstraint, elements=True)
    shaped.set_spherical_shape(10.0)
    shaped.compute_error(chain(), all8)
    assert np.linalg.norm(shaped.computed) <= np.linalg.norm(plain.computed) + 1e-9


def test_python_shape_function_zeros_high_r() -> None:
    shaped = pair_function(rmc.PairDistributionConstraint, 1.0, elements=True)
    shaped.set_shape_function(lambda r: 1.0 if r < 2.0 else 0.0)
    shaped.compute_error(chain(n=8, spacing=1.0), list(range(8)))
    centres = 0.1 * np.arange(50) + 0.05  # bin centres from r_min = 0
    assert np.all(shaped.computed[centres > 2.1] == 0.0)


def test_structure_factor_forms_differ() -> None:
    q = np.linspace(0.5, 10.0, 40)
    target = np.column_stack([q, np.ones_like(q)])
    atoms = rmc.AtomicStructure(chain(), ["C"] * 8)
    errors = []
    for kind in (rmc.StructureFactorConstraint, rmc.ReducedStructureFactorConstraint):
        c = kind()
        c.set_experimental_data(target)
        c.set_number_density(0.03)
        c.set_elements(atoms)
        c.initialise()
        errors.append(c.compute_error(chain(), list(range(8))))
        assert c.computed.shape == (40,)
    assert all(math.isfinite(e) for e in errors)
    assert errors[0] != errors[1]


def test_angular_distribution_knobs() -> None:
    adf = rmc.AngularDistributionConstraint()
    adf.set_cutoff(3.0)
    adf.set_smoothing(0)
    adf.set_scale_invariant(False)
    adf.set_resync_interval(100)
    assert adf.is_singular and not adf.is_rigid


# ---- cluster correlations ----


def ring_orbit() -> rmc.ClusterOrbit:
    orbit = rmc.ClusterOrbit(target=0.0, weight=1.0)
    for sites in ([0, 1], [1, 2], [2, 3], [3, 0]):
        orbit.add_instance(sites)
    return orbit


def test_cluster_orbit_is_a_described_struct() -> None:
    orbit = ring_orbit()
    assert orbit.body == 2
    assert orbit.instance_count == 4
    assert orbit.instances == [[0, 1], [1, 2], [2, 3], [3, 0]]
    assert orbit.flat_sites == [0, 1, 1, 2, 2, 3, 3, 0]
    assert repr(orbit).startswith("ClusterOrbit(body=2, flat_sites=[0, 1, 1, 2")
    assert orbit == ring_orbit()


def test_cluster_correlation_of_an_alternating_ring() -> None:
    atoms = rmc.AtomicStructure(np.zeros((4, 3)), ["Cu", "Au", "Cu", "Au"])
    c = rmc.ClusterCorrelationConstraint(atoms, {"Cu": 1.0, "Au": -1.0}, [ring_orbit()])
    assert c.current_correlations() == pytest.approx([-1.0])
    assert c.orbits[0].instance_count == 4


# ---- the holder, Python-defined constraints, the collection ----


class Pull:
    """Error = x_0 squared; records how it is called."""

    name = "Pull"
    cost = 0.5

    def __init__(self) -> None:
        self.calls = 0
        self.bc: rmc.BoundaryConditions | None = None
        self.initialised = False

    def compute_error(self, coords: np.ndarray, moved: np.ndarray) -> float:
        self.calls += 1
        assert not coords.flags.writeable
        assert moved.dtype == np.uint64
        return float(coords[0, 0] ** 2)

    def set_boundary_conditions(self, bc: rmc.BoundaryConditions) -> None:
        self.bc = bc

    def initialise(self) -> None:
        self.initialised = True


def test_python_constraint_round_trip() -> None:
    impl = Pull()
    held = rmc.Constraint(impl)
    assert held.name == "Pull"
    assert held.computation_cost == 0.5
    assert not held.is_rigid and not held.is_singular
    held.compute_before_move(pair(1.0, 0.0), ALL2)
    held.compute_after_move(pair(3.0, 0.0), ALL2)
    assert held.standard_error_before() == 1.0
    assert held.standard_error() == 9.0
    assert held.should_reject()
    held.accept()
    assert held.standard_error_before() == 9.0
    assert held.concrete() is impl
    assert impl.calls == 2


def test_python_constraint_hooks_are_forwarded() -> None:
    impl = Pull()
    held = rmc.Constraint(impl)
    held.set_boundary_conditions(rmc.PeriodicBC(5.0 * np.eye(3)))
    held.initialise()
    assert impl.bc is not None and impl.bc.periodic
    assert impl.initialised


def test_rigid_python_constraint_adds_nothing() -> None:
    impl = Pull()
    impl.rigid = True  # type: ignore[attr-defined]
    held = rmc.Constraint(impl)
    held.compute_before_move(pair(1.0, 0.0), ALL2)
    held.compute_after_move(pair(3.0, 0.0), ALL2)
    assert held.is_rigid
    assert held.standard_error() == 0.0
    assert held.should_reject()


def test_python_constraint_default_name_is_its_class() -> None:
    class Flat:
        def compute_error(self, coords: np.ndarray, moved: np.ndarray) -> float:
            return 0.0

    assert rmc.Constraint(Flat()).name == "Flat"


def test_python_constraint_needs_compute_error() -> None:
    with pytest.raises(TypeError, match="compute_error"):
        rmc.Constraint(object())


def test_python_exceptions_propagate() -> None:
    class Broken:
        def compute_error(self, coords: np.ndarray, moved: np.ndarray) -> float:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        rmc.Constraint(Broken()).compute_before_move(pair(0.0, 1.0), ALL2)


def test_collection_orders_cheapest_first_and_converts_implicitly() -> None:
    constraints = rmc.ConstraintCollection()
    constraints.add(pair_function(rmc.PairDistributionConstraint))
    bond = rmc.BondConstraint()
    bond.add_bond(0, 1, 1.0, 2.0)
    constraints.add(bond)
    assert len(constraints) == 2
    assert [c.name for c in constraints] == ["BondConstraint", "PairDistributionConstraint"]
    assert isinstance(constraints[1].concrete(), rmc.PairDistributionConstraint)
    assert [name for name, _ in constraints.error_breakdown()] == [
        "BondConstraint",
        "PairDistributionConstraint",
    ]
    with pytest.raises(IndexError):
        constraints[2]


def test_collection_refuses_a_second_singular_constraint() -> None:
    constraints = rmc.ConstraintCollection()
    constraints.add(rmc.PairDistributionConstraint())
    with pytest.raises(ValueError, match="singular"):
        constraints.add(rmc.PairDistributionConstraint())
    constraints.add(rmc.BondConstraint())
    constraints.add(rmc.BondConstraint())  # rigid, not singular: any number
    assert len(constraints) == 3


def test_concrete_is_a_reference_into_the_holder() -> None:
    constraints = rmc.ConstraintCollection()
    constraints.add(rmc.BondConstraint())
    inner = constraints[0].concrete()
    inner.add_bond(0, 1, 1.5, 2.5)
    assert constraints[0].concrete().compute_error(pair(0.0, 1.0), ALL2) == pytest.approx(0.5)
