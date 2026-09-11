"""Ensembles (replicas on threads, the GIL handed between them) and the
RMC_run pipeline as functions over RMCConfig."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import rmc


def chain(n: int, spacing: float) -> rmc.AtomicStructure:
    coords = np.zeros((n, 3))
    coords[:, 0] = spacing * np.arange(n)
    return rmc.AtomicStructure(coords, ["Ar"] * n)


class Squeeze:
    """A Python constraint, so every replica needs the GIL: error = spread in x."""

    def compute_error(self, coords: np.ndarray, moved: np.ndarray) -> float:
        return float(np.ptp(coords[:, 0]))


class FailsLater:
    def __init__(self) -> None:
        self.calls = 0

    def compute_error(self, coords: np.ndarray, moved: np.ndarray) -> float:
        self.calls += 1
        if self.calls > 20:
            raise RuntimeError("replica failed")
        return 0.0


def factory(created: list[rmc.Engine], constraint: type = Squeeze) -> Callable[[int], rmc.Engine]:
    def make(i: int) -> rmc.Engine:
        engine = rmc.Engine(chain(6, 2.5), rmc.InfiniteBC(1000.0))
        engine.build_atomic_groups(0.0, 0.3, seed=100 + i)
        engine.add_constraint(rmc.Constraint(constraint()))
        engine.set_track_best()
        created.append(engine)
        return engine

    return make


def test_run_ensemble_returns_the_best_replica_itself() -> None:
    created: list[rmc.Engine] = []
    winner = rmc.run_ensemble(factory(created), 3, 200)
    assert len(created) == 3
    assert any(winner is e for e in created)
    assert all(e.steps_total == 200 for e in created)
    assert winner.best_error == min(e.best_error for e in created)
    assert all(e.best_error < 12.5 for e in created)  # every replica squeezed


def test_prepare_runs_on_each_replica_before_it_runs() -> None:
    seen: list[int] = []
    rmc.run_ensemble(factory([]), 2, 10, prepare=lambda e: seen.append(e.steps_total))
    assert seen == [0, 0]


def test_an_exception_in_a_replica_surfaces() -> None:
    with pytest.raises(RuntimeError, match="replica failed"):
        rmc.run_ensemble(factory([], FailsLater), 2, 100)


def test_replica_preconditions() -> None:
    with pytest.raises(ValueError, match="n_replicas"):
        rmc.run_ensemble(factory([]), 0, 10)
    engine = factory([])(0)
    with pytest.raises(ValueError, match="same engine twice"):
        rmc.run_ensemble(lambda i: engine, 2, 10)
    with pytest.raises(ValueError, match="no groups"):
        rmc.run_ensemble(lambda i: rmc.Engine(chain(2, 1.0), rmc.InfiniteBC()), 1, 10)
    with pytest.raises(TypeError, match="must return an Engine, not str"):
        rmc.run_ensemble(lambda i: "not an engine", 1, 10)


def test_cooperative_ensemble_terminates_on_the_best_replica() -> None:
    created: list[rmc.Engine] = []
    winner = rmc.run_ensemble_cooperative(
        factory(created), 3, target_chi2=-1.0, sync_every=50, max_steps=200
    )
    assert any(winner is e for e in created)
    assert all(e.steps_total == 200 for e in created)
    assert winner.total_error == min(e.total_error for e in created)
    with pytest.raises(ValueError, match="sync_every"):
        rmc.run_ensemble_cooperative(factory([]), 2, 0.0, sync_every=0)


# ---- the RMC_run pipeline ----


def test_config_is_a_described_struct_with_the_cli_defaults() -> None:
    config = rmc.RMCConfig()
    assert (config.rho0, config.steps, config.seed) == (0.1, 100000, 42)
    assert config.move_gen == rmc.MoveGenKind.Random
    assert config.out_path == "refined.pdb"
    assert config.pdb_path == ""
    assert rmc.RMCConfig(seed=7) == rmc.RMCConfig(seed=7) != config
    assert repr(config).startswith("RMCConfig(pdb_path='', lammps_path=''")


@pytest.fixture
def inputs(tmp_path: Path) -> rmc.RMCConfig:
    start = rmc.make_random_amorphous(["Cu", "Zr"], [16, 16], seed=3)
    pdb = tmp_path / "start.pdb"
    rmc.write_pdb(start.structure, pdb)
    r = np.linspace(0.5, 8.0, 60)
    gr = tmp_path / "gr.dat"
    np.savetxt(gr, np.column_stack([r, np.ones_like(r)]))
    side = start.box[0, 0]
    return rmc.RMCConfig(
        pdb_path=str(pdb), pdf_path=str(gr), box_override=f"{side} {side} {side}",
        rho0=0.05, seed=3,
    )


def test_build_engine_from_a_config(inputs: rmc.RMCConfig) -> None:
    engine = rmc.build_engine(inputs)
    assert engine.n_groups == 32
    assert [c.name for c in engine.constraints] == ["PairDistributionConstraint"]
    assert engine.boundary.periodic
    engine.run(50)
    assert engine.steps_total == 50


def test_build_engine_from_loaded_inputs(inputs: rmc.RMCConfig) -> None:
    loaded = rmc.load_structure(inputs)
    data = rmc.load_experimental_data(inputs)
    assert data.pdf is not None and data.pdf.shape == (60, 2)
    assert data.sq is None and data.adf is None
    assert data == rmc.load_experimental_data(inputs)
    engines = [rmc.build_engine(loaded, data, rmc.RMCConfig(**{**_fields(inputs), "seed": s})) for s in (1, 2)]
    assert all(len(e.structure) == 32 for e in engines)


def _fields(config: rmc.RMCConfig) -> dict[str, object]:
    return {name: getattr(config, name) for name in ("pdb_path", "pdf_path", "box_override", "rho0")}


def test_gradient_move_generator_is_applied(inputs: rmc.RMCConfig) -> None:
    inputs.move_gen = rmc.MoveGenKind.Langevin
    inputs.move_step = 0.01
    engine = rmc.build_engine(inputs)
    rmc.apply_move_generator(engine, inputs)
    engine.run(5)
    assert np.isfinite(engine.structure.coordinates).all()


def test_config_errors_are_config_errors() -> None:
    with pytest.raises(rmc.ConfigError, match="exactly one input structure"):
        rmc.build_engine(rmc.RMCConfig())


@pytest.mark.parametrize("name", ["out.vasp", "out.lammps", "out.pdb"])
def test_write_structure_by_ext(cube_bc: rmc.PeriodicBC, name: str, tmp_path: Path) -> None:
    atoms = rmc.make_random_amorphous(["Cu"], [8], seed=1).structure
    box = rmc.periodic_box_or_zero(cube_bc)
    path = tmp_path / name
    rmc.write_structure_by_ext(atoms, box, path)
    assert len(rmc.read_structure_by_ext(path, type_to_element=["Cu"]).structure) == 8


def test_periodic_box_or_zero(cube_bc: rmc.PeriodicBC) -> None:
    np.testing.assert_allclose(rmc.periodic_box_or_zero(cube_bc), 10.0 * np.eye(3))
    np.testing.assert_allclose(rmc.periodic_box_or_zero(rmc.InfiniteBC()), np.zeros((3, 3)))
