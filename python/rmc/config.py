"""RMC_run's configuration as a validated model, and the refinement it drives."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import _core
from ._core import (
    AnnealingSampler,
    AtomicStructure,
    Chi2CollectorCallback,
    DirectionalOrderSelector,
    Engine,
    GreedySampler,
    MetropolisSampler,
    MoveGenKind,
    OrderedSelector,
    RandomSelector,
    RecursiveGroupSelector,
    SmartRandomSelector,
    WeightedRandomSelector,
)

__all__ = ["RMCConfig", "refine", "refine_ensemble"]

# The CLI's defaults, read off the C++ struct RMC_run binds its options to, so
# the two cannot drift.
_DEFAULTS: _core.RMCConfig = _core.RMCConfig()
_STRUCTURE_FIELDS = ("pdb_path", "lammps_path", "vasp_path")
_SEEDS = 2**32

Sampler = GreedySampler | MetropolisSampler | AnnealingSampler
Selector = (
    RandomSelector
    | WeightedRandomSelector
    | OrderedSelector
    | SmartRandomSelector
    | DirectionalOrderSelector
    | RecursiveGroupSelector
)
StepCallback = Callable[[int, int, int, float, AtomicStructure], None]


class RMCConfig(BaseModel):
    """The RMC_run knobs, validated. Exactly one of pdb_path, lammps_path and
    vasp_path names the starting structure; the other paths are optional."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pdb_path: Path | None = None
    lammps_path: Path | None = None
    """A LAMMPS data file (atom_style atomic); supplies the periodic box."""
    vasp_path: Path | None = None
    """A VASP5 POSCAR/CONTCAR; supplies the periodic cell."""
    lammps_types: tuple[str, ...] = tuple(_DEFAULTS.lammps_types)
    """Element symbols for LAMMPS atom types 1, 2, ..."""
    box_override: tuple[float, float, float] | Literal["inf"] | None = None
    """Replace the input's cell: (a, b, c) orthogonal periodic, or "inf"."""

    pdf_path: Path | None = None
    """Experimental G(r), two columns."""
    sq_path: Path | None = None
    """Experimental S(Q), two columns."""
    adf_path: Path | None = None
    """Experimental bond-angle distribution, multi-column."""

    rho0: float = Field(_DEFAULTS.rho0, gt=0)
    """Number density (atoms per cubic Angstrom)."""
    adf_cutoff: float = Field(_DEFAULTS.adf_cutoff, gt=0)
    adf_smooth: int = Field(_DEFAULTS.adf_smooth, ge=0)
    steps: int = Field(_DEFAULTS.steps, ge=1)
    seed: int = Field(_DEFAULTS.seed, ge=0, lt=_SEEDS)
    use_smart: bool = _DEFAULTS.use_smart
    """Use the adaptive SmartRandomSelector."""
    group_min_amp: float = Field(_DEFAULTS.group_min_amp, ge=0)
    group_max_amp: float = Field(_DEFAULTS.group_max_amp, ge=0)
    """Per-atom translation range (Angstrom)."""
    move_gen: MoveGenKind = _DEFAULTS.move_gen
    move_step: float = Field(_DEFAULTS.move_step, gt=0)
    """Gradient step (Angstrom) for the Langevin and Leapfrog movers."""
    checkpoint_path: Path | None = None
    log_every: int = Field(_DEFAULTS.log_every, ge=1)
    out_path: Path = Path(_DEFAULTS.out_path)
    """Where refine() writes the result; the extension picks the format."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        given = [name for name in _STRUCTURE_FIELDS if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError(
                f"exactly one of {', '.join(_STRUCTURE_FIELDS)} must be set, not {len(given)}"
            )
        if self.group_min_amp > self.group_max_amp:
            raise ValueError("group_min_amp must not exceed group_max_amp")
        return self

    def to_core(self) -> _core.RMCConfig:
        """The C++ struct the builder functions take."""
        core = _core.RMCConfig()
        for name in type(self).model_fields:
            setattr(core, name, _core_value(name, getattr(self, name)))
        return core


def _core_value(name: str, value: object) -> object:
    match value:
        case None:
            return ""  # the C++ struct's "not supplied"
        case Path():
            return str(value)
        case tuple() if name == "box_override":
            return " ".join(map(str, value))
        case tuple():
            return list(value)
        case _:
            return value


def _fan_out(callbacks: Sequence[StepCallback]) -> StepCallback:
    if len(callbacks) == 1:
        return callbacks[0]

    def every(step: int, accepted: int, tried: int, chi2: float, s: AtomicStructure) -> None:
        for callback in callbacks:
            callback(step, accepted, tried, chi2, s)

    return every


def _write_result(engine: Engine, config: RMCConfig) -> None:
    box = _core.periodic_box_or_zero(engine.boundary)
    _core.write_structure_by_ext(engine.structure, box, config.out_path)


def refine(
    config: RMCConfig,
    *,
    sampler: Sampler | None = None,
    selector: Selector | None = None,
    step_callback: StepCallback | None = None,
    chi2_csv: str | os.PathLike[str] | None = None,
) -> Engine:
    """Refine as RMC_run does: build the engine config names, run config.steps
    steps and write config.out_path. sampler and selector replace the
    defaults; step_callback runs every config.log_every steps; chi2_csv
    records the chi^2 history there. Returns the engine, to read its
    constraints and statistics."""
    core = config.to_core()
    engine = _core.build_engine(core)
    _core.apply_move_generator(engine, core)
    if sampler is not None:
        engine.set_sampler(sampler, seed=config.seed)
    if selector is not None:
        engine.set_selector(selector)
    if config.checkpoint_path is not None:
        engine.set_checkpoint(config.checkpoint_path)

    with contextlib.ExitStack() as stack:
        callbacks = [step_callback] if step_callback is not None else []
        if chi2_csv is not None:
            callbacks.append(stack.enter_context(Chi2CollectorCallback(Path(chi2_csv))))
        if callbacks:
            engine.set_step_callback(_fan_out(callbacks), log_every=config.log_every)
        engine.run(config.steps)
        engine.clear_step_callback()

    _write_result(engine, config)
    return engine


def refine_ensemble(config: RMCConfig, n_replicas: int, *, tbb_threads: int = 0) -> Engine:
    """RMC_run --ensemble: n_replicas engines from one read of the inputs,
    replica i seeded config.seed + i, run in parallel; the one with the lowest
    error is written to config.out_path and returned."""
    core = config.to_core()
    loaded = _core.load_structure(core)
    data = _core.load_experimental_data(core)

    def make(replica: int) -> Engine:
        seeded = config.model_copy(update={"seed": (config.seed + replica) % _SEEDS})
        return _core.build_engine(loaded, data, seeded.to_core())

    winner = _core.run_ensemble(
        make,
        n_replicas,
        config.steps,
        tbb_threads=tbb_threads,
        prepare=lambda engine: _core.apply_move_generator(engine, core),
    )
    _write_result(winner, config)
    return winner
