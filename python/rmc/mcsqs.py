"""The SQS (special quasirandom structure) search.

:mod:`rmc._core.mcsqs` parses ATAT lattices and enumerates a supercell's
cluster orbits on seitz; this module adds the engine set-up ``mcsqs_rmc``
uses, so a search is three calls::

    from rmc import mcsqs

    sqs = mcsqs.enumerate(mcsqs.parse_lattice("rndstr.in"), [[2, 0, 0], [0, 2, 0], [0, 0, 2]], {2: 3.0})
    engine = mcsqs.sqs_engine(sqs, seed=7)
    engine.run(100_000)
"""

from __future__ import annotations

from ._core import (
    AnnealingSampler,
    AtomicStructure,
    ClusterCorrelationConstraint,
    Engine,
    GreedySampler,
    Group,
    InfiniteBC,
    MetropolisSampler,
    SmartRandomSelector,
    SpeciesSwapGenerator,
)
from ._core.mcsqs import (
    AtatLattice,
    EnumeratedSqs,
    LatticeSite,
    enumerate,
    parse_lattice,
    parse_lattice_text,
    write_str_out,
)

__all__ = sorted(
    [
        "AtatLattice",
        "EnumeratedSqs",
        "LatticeSite",
        "build_sublattices",
        "enumerate",
        "parse_lattice",
        "parse_lattice_text",
        "sqs_engine",
        "write_str_out",
    ]
)

Sampler = GreedySampler | MetropolisSampler | AnnealingSampler


def build_sublattices(structure: AtomicStructure) -> list[list[int]]:
    """Site indices grouped by residue name (enumerate labels sublattices
    SL0, SL1, ...), groups in order of first appearance."""
    groups: dict[str, list[int]] = {}
    for site, residue in zip(range(len(structure)), structure.residues, strict=True):
        groups.setdefault(residue, []).append(site)
    return list(groups.values())


def sqs_engine(sqs: EnumeratedSqs, *, sampler: Sampler | None = None, seed: int = 42) -> Engine:
    """An engine that searches sqs's supercell for the SQS, set up as
    ``mcsqs_rmc`` does: the cluster-correlation constraint, one species-swap
    group per site (swaps stay on their sublattice), a SmartRandomSelector and
    best-state tracking. sampler defaults to GreedySampler(); ``mcsqs_rmc``
    anneals, which AnnealingSampler reproduces. Read the result from
    engine.best_structure."""
    engine = Engine(sqs.structure, InfiniteBC())
    engine.add_constraint(
        ClusterCorrelationConstraint(engine.structure, sqs.occ_index, sqs.table, sqs.orbits)
    )
    swap = SpeciesSwapGenerator(engine.structure, build_sublattices(engine.structure), seed=seed)
    for site in range(len(engine.structure)):
        engine.add_group(Group(f"site_{site}", [site], swap))
    engine.set_selector(SmartRandomSelector(seed=seed))
    engine.set_sampler(sampler if sampler is not None else GreedySampler(), seed=seed)
    engine.set_track_best()
    return engine
