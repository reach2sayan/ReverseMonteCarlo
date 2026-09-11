"""Small structures shared by the suites."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rmc


@pytest.fixture
def square() -> rmc.AtomicStructure:
    """Four atoms on a unit square in the z = 0 plane, alternating Cu/Zr."""
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    return rmc.AtomicStructure(coords, ["Cu", "Zr", "Cu", "Zr"])


@pytest.fixture
def cube_bc() -> rmc.PeriodicBC:
    """A 10 Angstrom cubic cell."""
    return rmc.PeriodicBC(10.0 * np.eye(3))


@pytest.fixture
def cuzr() -> rmc.RandomStructure:
    """64 atoms of Cu50Zr50, seeded."""
    return rmc.make_random_amorphous(["Cu", "Zr"], [32, 32], seed=7)


@pytest.fixture
def refinement_files(tmp_path: Path) -> tuple[Path, Path, float]:
    """A 32-atom Cu/Zr start written as PDB, a flat G(r) target, and the cell side."""
    start = rmc.make_random_amorphous(["Cu", "Zr"], [16, 16], seed=3)
    pdb = tmp_path / "start.pdb"
    rmc.write_pdb(start.structure, pdb)
    r = np.linspace(0.5, 8.0, 60)
    gr = tmp_path / "gr.dat"
    np.savetxt(gr, np.column_stack([r, np.ones_like(r)]))
    return pdb, gr, float(start.box[0, 0])
