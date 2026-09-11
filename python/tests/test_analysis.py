"""g(r) and the bond-angle distribution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rmc

GR = rmc.GrParams(r_max=6.0, n_bins=60)


def test_params_are_described_structs() -> None:
    gr = rmc.GrParams()
    assert (gr.r_min, gr.r_max, gr.n_bins, gr.exclude_intra) == (0.0, 10.0, 200, False)
    adf = rmc.AdfParams()
    assert (adf.max_dis, adf.n_bins, adf.smooth_range) == (3.4, 100, 2)
    assert rmc.GrParams(n_bins=5) == rmc.GrParams(n_bins=5)


def test_gr_of_a_random_structure(cuzr: rmc.RandomStructure) -> None:
    g = rmc.compute_gr(cuzr.structure, cuzr.periodic_bc(), params=GR)
    assert g.r.shape == g.total.shape == (60,)
    assert not g.r.flags.writeable
    assert len(g.partials) == 3  # Cu-Cu, Cu-Zr, Zr-Zr
    assert all(c.values.shape == (60,) for c in g.partials)
    assert [(s.symbol, s.count) for s in g.species] == [("Cu", 32), ("Zr", 32)]
    assert g.density == pytest.approx(64 / np.linalg.det(cuzr.box))
    # Grid pitch 3 A on a body-centred lattice: no pair is closer than 2 A.
    assert np.all(g.total[g.r < 2.0] == 0.0)
    assert np.isfinite(g.total).all() and g.total.max() > 0.0
    columns = g.as_dict()
    assert list(columns)[:2] == ["r", "total"]
    assert set(columns) == {"r", "total", *(c.label for c in g.partials)}


def test_gr_from_a_file_matches(cuzr: rmc.RandomStructure, tmp_path: Path) -> None:
    path = tmp_path / "POSCAR"
    rmc.write_vasp(cuzr.structure, cuzr.box, path)
    from_file = rmc.compute_gr(path, rmc.InfiniteBC(), params=GR)
    in_memory = rmc.compute_gr(cuzr.structure, cuzr.periodic_bc(), params=GR)
    np.testing.assert_allclose(from_file.total, in_memory.total, atol=1e-6)


def test_gr_rejects_an_empty_range(cuzr: rmc.RandomStructure) -> None:
    with pytest.raises(rmc.AnalysisError, match="r_max > r_min"):
        rmc.compute_gr(cuzr.structure, cuzr.periodic_bc(), params=rmc.GrParams(r_min=5.0, r_max=1.0))


def test_write_gr(cuzr: rmc.RandomStructure, tmp_path: Path) -> None:
    path = tmp_path / "gr.dat"
    rmc.write_gr(rmc.compute_gr(cuzr.structure, cuzr.periodic_bc(), params=GR), path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("#")
    assert sum(not line.startswith("#") for line in lines) == 60


def test_adf_of_a_random_structure(cuzr: rmc.RandomStructure, tmp_path: Path) -> None:
    a = rmc.compute_adf(cuzr.structure, cuzr.periodic_bc(), params=rmc.AdfParams(n_bins=36))
    assert a.theta.shape == a.total.shape == (36,)
    assert np.isfinite(a.total).all()
    assert a.max_dis == pytest.approx(3.4)
    assert [s.symbol for s in a.species] == ["Cu", "Zr"]
    path = tmp_path / "adf.dat"
    rmc.write_adf(a, path)
    assert path.read_text().startswith("#")
