"""The engine, mirroring tests/test_engine.cpp, plus the binding's guarantees:
preconditions raise, callbacks borrow safely, curves are read back."""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
import pytest

import rmc


def chain(n: int, spacing: float) -> rmc.AtomicStructure:
    coords = np.zeros((n, 3))
    coords[:, 0] = spacing * np.arange(n)
    return rmc.AtomicStructure(coords, ["Ar"] * n, molecule_ids=list(range(n)))


def test_constructs_and_runs() -> None:
    engine = rmc.Engine(chain(5, 3.0), rmc.InfiniteBC(1000.0))
    engine.build_atomic_groups(0.0, 0.1, seed=42)
    engine.run(100)
    assert engine.stats.steps_total == engine.steps_total == 100
    assert engine.stats.steps_tried <= 100
    assert engine.n_groups == 5
    assert repr(engine) == "Engine(5 atoms, 5 groups, 0 constraints, 100 steps)"


def test_unconstrained_moves_are_all_accepted() -> None:
    engine = rmc.Engine(chain(10, 3.0), rmc.InfiniteBC(1e6))
    engine.build_atomic_groups(0.05, 0.05, seed=1)
    engine.run(500)
    assert engine.stats.steps_accepted == engine.stats.steps_tried


def test_hard_distance_constraint_limits_proximity() -> None:
    engine = rmc.Engine(chain(3, 4.0), rmc.InfiniteBC(1e6))
    engine.build_atomic_groups(0.0, 0.5, seed=99)
    close = rmc.InterMolecularDistanceConstraint()
    close.set_minimum_distance("Ar", "Ar", 2.0)
    close.set_structure(engine.structure)
    engine.add_constraint(close)
    engine.run(2000)
    coords = engine.structure.coordinates
    gaps = [np.linalg.norm(coords[i] - coords[j]) for i in range(3) for j in range(i + 1, 3)]
    assert min(gaps) >= 2.0 - 1e-3


def test_rejected_moves_restore_the_snapshot() -> None:
    engine = rmc.Engine(chain(2, 1.0), rmc.InfiniteBC(1e6))
    engine.build_atomic_groups(0.5, 0.5, seed=7)
    bond = rmc.BondConstraint()
    bond.add_bond(0, 1, 1.0, 1.0)
    engine.add_constraint(bond)
    before = engine.structure.coordinates.copy()
    engine.run(100)
    np.testing.assert_allclose(engine.structure.coordinates, before, atol=1e-10)
    assert engine.steps_accepted == 0


def test_periodic_wrapping_keeps_atoms_in_the_box() -> None:
    engine = rmc.Engine(chain(4, 3.0), rmc.PeriodicBC(12.0 * np.eye(3)))
    engine.build_atomic_groups(0.0, 0.5, seed=55)
    engine.run(500)
    x = engine.structure.coordinates[:, 0]
    assert (x >= -1e-6).all() and (x < 12.0 + 1e-6).all()
    assert engine.boundary.periodic


def test_run_without_groups_raises() -> None:
    engine = rmc.Engine(chain(2, 1.0), rmc.InfiniteBC())
    with pytest.raises(ValueError, match="no groups"):
        engine.run(10)
    with pytest.raises(ValueError, match="no groups"):
        engine.run_until(0.0)


def test_engine_copies_its_structure() -> None:
    atoms = chain(3, 1.0)
    engine = rmc.Engine(atoms, rmc.InfiniteBC())
    atoms.coordinates[0, 0] = 99.0
    assert engine.structure.coordinates[0, 0] == 0.0


def test_structure_is_a_live_reference() -> None:
    engine = rmc.Engine(chain(3, 1.0), rmc.InfiniteBC())
    view = engine.structure.coordinates
    engine.build_atomic_groups(0.1, 0.1, seed=3)
    engine.run(30)
    np.testing.assert_array_equal(view, engine.structure.coordinates)
    assert not np.array_equal(view[:, 0], [0.0, 1.0, 2.0])


def test_duplicate_singular_constraint_raises() -> None:
    engine = rmc.Engine(chain(3, 1.0), rmc.InfiniteBC())
    engine.add_constraint(rmc.PairDistributionConstraint())
    with pytest.raises(ValueError, match="singular"):
        engine.add_constraint(rmc.PairDistributionConstraint())


def test_computed_curve_is_read_through_engine_constraints() -> None:
    engine = rmc.Engine(chain(8, 3.0), rmc.InfiniteBC(1000.0))
    engine.build_atomic_groups(0.0, 0.2, seed=4)
    pdf = rmc.PairDistributionConstraint()
    r = 0.1 * np.arange(1, 51)
    pdf.set_experimental_data(np.column_stack([r, np.ones_like(r)]))
    pdf.set_number_density(0.03)
    pdf.set_elements(engine.structure)
    engine.add_constraint(pdf)
    engine.run(50)
    held = engine.constraints[0].concrete()
    assert isinstance(held, rmc.PairDistributionConstraint)
    assert held.computed.shape == (50,)
    assert np.linalg.norm(held.computed) > 0.0
    assert pdf.computed.shape == (0,) or not np.array_equal(pdf.computed, held.computed)
    assert engine.total_error == pytest.approx(engine.constraints.total_error)


def test_step_callback_borrows_the_structure_safely() -> None:
    seen: list[tuple[int, rmc.AtomicStructure]] = []

    def record(step: int, accepted: int, tried: int, chi2: float, s: rmc.AtomicStructure) -> None:
        assert accepted <= tried <= step
        seen.append((step, s))

    engine = rmc.Engine(chain(3, 2.0), rmc.InfiniteBC())
    engine.build_atomic_groups(0.0, 0.1, seed=5)
    engine.set_step_callback(record, log_every=10)
    engine.run(35)
    assert [step for step, _ in seen] == [10, 20, 30]
    kept = seen[-1][1]
    del engine
    gc.collect()
    assert len(kept) == 3  # the kept structure holds the engine alive
    with pytest.raises(ValueError):
        rmc.Engine(chain(1, 1.0), rmc.InfiniteBC()).set_step_callback(record, log_every=0)


def test_best_structure_is_an_independent_copy() -> None:
    engine = rmc.Engine(chain(4, 2.0), rmc.InfiniteBC())
    engine.build_atomic_groups(0.0, 0.2, seed=6)
    engine.set_track_best()
    engine.run(20)
    best = engine.best_structure
    best.coordinates[:] = 99.0
    assert (engine.structure.coordinates != 99.0).any()
    assert (engine.best_structure.coordinates != 99.0).all()


def test_python_constraint_runs_inside_the_engine() -> None:
    class Centre:
        """Pull atom 0 towards x = 0."""

        def __init__(self) -> None:
            self.calls = 0

        def compute_error(self, coords: np.ndarray, moved: np.ndarray) -> float:
            self.calls += 1
            return float(coords[0, 0] ** 2)

    impl = Centre()
    atoms = chain(2, 3.0)
    atoms.coordinates[0, 0] = 2.0
    engine = rmc.Engine(atoms, rmc.InfiniteBC())
    engine.build_atomic_groups(0.0, 0.3, seed=8)
    engine.add_constraint(rmc.Constraint(impl))
    engine.run(300)
    assert impl.calls > 0
    assert abs(engine.structure.coordinates[0, 0]) < 2.0
    assert engine.constraints[0].concrete() is impl


def test_python_exception_inside_run_propagates() -> None:
    class Broken:
        def __init__(self) -> None:
            self.calls = 0

        def compute_error(self, coords: np.ndarray, moved: np.ndarray) -> float:
            self.calls += 1
            if self.calls > 3:
                raise RuntimeError("constraint failed")
            return 0.0

    engine = rmc.Engine(chain(2, 3.0), rmc.InfiniteBC())
    engine.build_atomic_groups(0.0, 0.1, seed=9)
    engine.add_constraint(rmc.Constraint(Broken()))
    with pytest.raises(RuntimeError, match="constraint failed"):
        engine.run(100)


def test_samplers_and_selectors_install() -> None:
    engine = rmc.Engine(chain(4, 2.0), rmc.InfiniteBC())
    engine.build_atomic_groups(0.0, 0.1, seed=10)
    for sampler in (
        rmc.GreedySampler(0.1),
        rmc.MetropolisSampler(2.0),
        rmc.AnnealingSampler(rmc.AnnealingSampler.Schedule(interval=5)),
    ):
        engine.set_sampler(sampler, seed=3)
        engine.run(10)
    for selector in (
        rmc.RandomSelector(seed=1),
        rmc.OrderedSelector(),
        rmc.SmartRandomSelector(),
        rmc.RecursiveGroupSelector(rmc.OrderedSelector()),
    ):
        engine.set_selector(selector)
        engine.run(10)
    assert engine.steps_total == 70


def test_removal_group_uses_the_engine_collector() -> None:
    engine = rmc.Engine(chain(4, 2.0), rmc.InfiniteBC())
    engine.add_removal_group("drop", [3])
    engine.run(5)
    assert engine.collector.is_removed(3)
    assert engine.collector.n_removed == 1
    assert engine.collector.n_active(4) == 3


def test_frames_and_frame_selector() -> None:
    engine = rmc.Engine(chain(3, 2.0), rmc.InfiniteBC())
    engine.add_frame(chain(3, 2.5))
    engine.set_frame_selector(rmc.OrderedSelector())
    engine.build_atomic_groups(0.0, 0.1, seed=11)
    engine.run(20)
    assert engine.steps_total == 20


def test_chi2_collector_as_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "chi2.csv"
    with rmc.Chi2CollectorCallback(path) as collector:
        engine = rmc.Engine(chain(3, 2.0), rmc.InfiniteBC())
        engine.build_atomic_groups(0.0, 0.1, seed=12)
        engine.add_constraint(rmc.Constraint(type("Zero", (), {"compute_error": lambda self, c, m: 1.0})()))
        engine.set_step_callback(collector, log_every=5)
        engine.run(20)
    assert [step for step, _ in collector.history] == [5, 10, 15, 20]
    rows = list(csv.reader(path.open()))
    assert rows[0] == ["step", "chi2"] and len(rows) == 5
    collector.finalize()  # idempotent


def test_pdb_snapshot_callback(tmp_path: Path) -> None:
    snapshots = rmc.PDBSnapshotCallback(tmp_path / "frames")
    engine = rmc.Engine(chain(3, 2.0), rmc.InfiniteBC())
    engine.build_atomic_groups(0.0, 0.1, seed=13)
    engine.set_step_callback(snapshots, log_every=10)
    engine.run(20)
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == [
        "step_0000000010.pdb",
        "step_0000000020.pdb",
    ]


def test_histogram_callback(tmp_path: Path) -> None:
    axis = np.array([1.0, 2.0, 3.0])
    hist = rmc.HistogramCallback(axis, lambda: (axis * 2.0, axis * 3.0), tmp_path)
    hist(7, 0, 0, 0.0, chain(1, 1.0))
    rows = list(csv.reader((tmp_path / "hist_0000000007.csv").open()))
    assert rows[0] == ["axis", "computed", "experimental"]
    assert rows[1] == ["1", "2", "3"]
