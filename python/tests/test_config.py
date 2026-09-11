"""The validated RMCConfig and the refinement it drives."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import rmc


def config_for(files: tuple[Path, Path, float], tmp_path: Path, **overrides: object) -> rmc.RMCConfig:
    pdb, gr, side = files
    fields: dict[str, object] = {
        "pdb_path": pdb,
        "pdf_path": gr,
        "box_override": (side, side, side),
        "rho0": 0.05,
        "steps": 200,
        "log_every": 50,
        "out_path": tmp_path / "refined.vasp",
    }
    return rmc.RMCConfig(**{**fields, **overrides})


def test_defaults_are_read_off_the_cli_struct() -> None:
    core = rmc._core.RMCConfig()
    config = rmc.RMCConfig(pdb_path="start.pdb")
    for name in ("rho0", "adf_cutoff", "adf_smooth", "steps", "seed", "use_smart",
                 "group_min_amp", "group_max_amp", "move_gen", "move_step", "log_every"):
        assert getattr(config, name) == getattr(core, name), name
    assert config.out_path == Path(core.out_path)


def test_the_model_is_frozen_and_closed() -> None:
    config = rmc.RMCConfig(pdb_path="start.pdb")
    with pytest.raises(ValidationError):
        config.steps = 5  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra"):
        rmc.RMCConfig(pdb_path="start.pdb", step=5)  # type: ignore[call-arg]


@pytest.mark.parametrize("paths", [{}, {"pdb_path": "a.pdb", "vasp_path": "POSCAR"}])
def test_exactly_one_structure(paths: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="exactly one of pdb_path"):
        rmc.RMCConfig(**paths)


@pytest.mark.parametrize(
    "bad",
    [
        {"rho0": 0.0},
        {"steps": 0},
        {"seed": -1},
        {"seed": 2**32},
        {"adf_smooth": -1},
        {"move_step": 0.0},
        {"log_every": 0},
        {"group_min_amp": 0.3, "group_max_amp": 0.2},
        {"box_override": "10 10 10"},
    ],
)
def test_ranges_are_checked(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        rmc.RMCConfig(pdb_path="start.pdb", **bad)


def test_to_core_converts_every_field() -> None:
    config = rmc.RMCConfig(
        lammps_path=Path("in.lmp"),
        lammps_types=("Zr", "Cu"),
        box_override=(10.0, 11.0, 12.5),
        pdf_path=Path("gr.dat"),
        move_gen=rmc.MoveGenKind.Langevin,
        seed=9,
    )
    core = config.to_core()
    assert isinstance(core, rmc._core.RMCConfig)
    assert (core.lammps_path, core.pdb_path, core.sq_path) == ("in.lmp", "", "")
    assert core.lammps_types == ["Zr", "Cu"]
    assert core.box_override == "10.0 11.0 12.5"
    assert core.pdf_path == "gr.dat"
    assert core.move_gen == rmc.MoveGenKind.Langevin
    assert core.seed == 9
    assert rmc.RMCConfig(pdb_path="a.pdb", box_override="inf").to_core().box_override == "inf"


def test_refine_runs_and_writes(refinement_files: tuple[Path, Path, float], tmp_path: Path) -> None:
    seen: list[int] = []
    config = config_for(refinement_files, tmp_path)
    engine = rmc.refine(
        config,
        step_callback=lambda step, *_: seen.append(step),
        chi2_csv=tmp_path / "chi2.csv",
    )
    assert engine.steps_total == 200
    assert seen == [50, 100, 150, 200]
    rows = list(csv.reader((tmp_path / "chi2.csv").open()))
    assert rows[0] == ["step", "chi2"] and len(rows) == 5
    written = rmc.read_vasp(config.out_path)
    assert len(written.structure) == 32
    np.testing.assert_allclose(written.box, np.diag([refinement_files[2]] * 3))


def test_refine_takes_a_sampler_and_selector(
    refinement_files: tuple[Path, Path, float], tmp_path: Path
) -> None:
    config = config_for(refinement_files, tmp_path, steps=50, out_path=tmp_path / "out.pdb")
    engine = rmc.refine(config, sampler=rmc.MetropolisSampler(0.1), selector=rmc.OrderedSelector())
    assert engine.steps_total == 50
    assert (tmp_path / "out.pdb").exists()


def test_refine_ensemble_writes_the_winner(
    refinement_files: tuple[Path, Path, float], tmp_path: Path
) -> None:
    config = config_for(refinement_files, tmp_path, steps=100)
    winner = rmc.refine_ensemble(config, 2)
    assert winner.steps_total == 100
    assert len(rmc.read_vasp(config.out_path).structure) == 32


def test_a_missing_input_is_a_config_error(tmp_path: Path) -> None:
    config = rmc.RMCConfig(pdb_path=tmp_path / "missing.pdb")
    with pytest.raises(rmc.ConfigError, match="Cannot open PDB file"):
        rmc.refine(config)


def test_errors_module_and_protocols() -> None:
    assert rmc.errors.IoError is rmc.IoError
    assert sorted(rmc.errors.__all__) == rmc.errors.__all__

    class Flat:
        def compute_error(self, coords: np.ndarray, moved: np.ndarray) -> float:
            return 0.0

    class Still:
        def generate(self, coords: np.ndarray, indices: np.ndarray) -> None:
            pass

    assert isinstance(Flat(), rmc.ConstraintProtocol)
    assert not isinstance(Still(), rmc.ConstraintProtocol)
    assert isinstance(Still(), rmc.MoveGeneratorProtocol)
    assert rmc.python_constraint(Flat()).name == "Flat"
    assert isinstance(rmc.python_generator(Still()), rmc.MoveGenerator)
