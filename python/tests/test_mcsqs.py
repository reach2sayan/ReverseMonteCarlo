"""The SQS pipeline, mirroring tests/test_atat_pipeline.cpp, and rmc.mcsqs's
engine set-up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rmc
from rmc import mcsqs

CUBIC = "1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 1 0\n0 0 1\n0 0 0 Cu=0.5,Au=0.5\n"
CHAIN = "1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 5 0\n0 0 5\n0 0 0 Cu=0.5,Au=0.5\n"
FCC = "1 0 0\n0 1 0\n0 0 1\n0 0.5 0.5\n0.5 0 0.5\n0.5 0.5 0\n0 0 0 Cu=0.5,Au=0.5\n"
TWO_SUBLATTICES = (
    "1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 1 0\n0 0 1\n"
    "0 0 0 Cu=0.5,Au=0.5\n0.5 0.5 0.5 Ni=0.5,Ti=0.5\n"
)


def diag(a: int, b: int, c: int) -> np.ndarray:
    return np.diag([a, b, c])


def test_parse_a_cubic_binary() -> None:
    lattice = mcsqs.parse_lattice_text(CUBIC)
    np.testing.assert_allclose(lattice.cell, np.eye(3))
    assert len(lattice.sites) == 1
    assert len(lattice.sites[0].occ) == 2
    assert lattice.labels == ["Au", "Cu"]
    assert (lattice.occupation_index("Au"), lattice.occupation_index("Cu")) == (0, 1)


def test_parse_a_file(tmp_path: Path) -> None:
    path = tmp_path / "rndstr.in"
    path.write_text(CUBIC)
    assert mcsqs.parse_lattice(path).labels == ["Au", "Cu"]


def test_malformed_lattice_is_an_mcsqs_error() -> None:
    with pytest.raises(rmc.McsqsError):
        mcsqs.parse_lattice_text("1 0 0\n")
    assert issubclass(rmc.McsqsError, rmc.RmcError)


def test_chain_pair_orbit() -> None:
    sqs = mcsqs.enumerate(mcsqs.parse_lattice_text(CHAIN), diag(4, 1, 1), {2: 1.1}, seed=1)
    assert len(sqs.structure) == 4
    assert len(sqs.orbits) == 1
    assert sqs.orbits[0].instance_count == 4
    assert sqs.orbits[0].target == pytest.approx(0.0, abs=1e-9)


def test_fcc_first_pair_orbit() -> None:
    sqs = mcsqs.enumerate(mcsqs.parse_lattice_text(FCC), diag(2, 2, 2), {2: 0.75}, seed=3)
    assert len(sqs.structure) == 8
    assert len(sqs.orbits) == 1
    assert sqs.orbits[0].instance_count == 48
    assert sqs.structure.elements.count("Cu") == 4


def test_two_sublattices() -> None:
    sqs = mcsqs.enumerate(
        mcsqs.parse_lattice_text(TWO_SUBLATTICES), diag(2, 2, 2), {2: 0.9}, seed=5
    )
    assert len(sqs.structure) == 16
    assert len(sqs.table) == 2
    sublattices = mcsqs.build_sublattices(sqs.structure)
    assert sorted(map(len, sublattices)) == [8, 8]
    for sites in sublattices:
        species = {sqs.structure.elements[i] for i in sites}
        assert species <= {"Cu", "Au"} or species <= {"Ni", "Ti"}
    assert sqs.orbits[0].instance_count == 64


def test_sqs_engine_searches_within_the_composition() -> None:
    sqs = mcsqs.enumerate(mcsqs.parse_lattice_text(FCC), diag(2, 2, 2), {2: 0.75}, seed=3)
    engine = mcsqs.sqs_engine(sqs, seed=3)
    engine.initialise()
    start = engine.total_error
    engine.run(500)
    assert engine.best_error <= start
    best = engine.best_structure
    assert best.elements.count("Cu") == 4
    assert engine.n_groups == 8


def test_sqs_engine_takes_a_sampler() -> None:
    sqs = mcsqs.enumerate(mcsqs.parse_lattice_text(FCC), diag(2, 2, 2), {2: 0.75}, seed=4)
    engine = mcsqs.sqs_engine(sqs, sampler=rmc.MetropolisSampler(0.5), seed=4)
    engine.run(100)
    assert engine.steps_total == 100


def test_write_str_out(tmp_path: Path) -> None:
    sqs = mcsqs.enumerate(mcsqs.parse_lattice_text(FCC), diag(2, 2, 2), {2: 0.75}, seed=3)
    path = tmp_path / "bestsqs.out"
    mcsqs.write_str_out(path, sqs.axes, sqs.supercell, sqs.frac_positions, sqs.structure.elements)
    assert len(path.read_text().splitlines()) >= 6 + 8


def test_the_submodule_is_importable() -> None:
    from rmc._core.mcsqs import enumerate as raw_enumerate

    assert mcsqs.enumerate is raw_enumerate
    assert mcsqs.__all__ == sorted(mcsqs.__all__)
    public = {name for name in dir(rmc._core.mcsqs) if not name.startswith("_")}
    assert public <= set(mcsqs.__all__)
    assert rmc.mcsqs is mcsqs


def test_lattice_site_is_a_described_struct() -> None:
    site = mcsqs.LatticeSite(frac=[0.5, 0.5, 0.5], occ=[("Ni", 0.5), ("Ti", 0.5)])
    assert site.occ == [("Ni", 0.5), ("Ti", 0.5)]
    assert site == mcsqs.LatticeSite(frac=[0.5, 0.5, 0.5], occ=[("Ni", 0.5), ("Ti", 0.5)])
