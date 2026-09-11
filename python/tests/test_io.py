"""Structure, data and checkpoint files round-trip through the readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rmc

BOX = np.diag([10.0, 11.0, 12.0])


def by_element(atoms: rmc.AtomicStructure) -> dict[str, np.ndarray]:
    """Coordinates per element, rows sorted: VASP regroups atoms by element."""
    elements = np.array(atoms.elements)
    return {
        e: np.array(sorted(map(tuple, atoms.coordinates[elements == e])))
        for e in sorted(set(atoms.elements))
    }


def test_pdb_round_trip(square: rmc.AtomicStructure, tmp_path: Path) -> None:
    path = tmp_path / "square.pdb"
    rmc.write_pdb(square, path)
    back = rmc.read_pdb(path)
    assert back.elements == square.elements
    np.testing.assert_allclose(back.coordinates, square.coordinates, atol=1e-3)


def test_paths_may_be_str(square: rmc.AtomicStructure, tmp_path: Path) -> None:
    path = str(tmp_path / "square.pdb")
    rmc.write_pdb(square, path)
    assert len(rmc.read_pdb(path)) == 4


def test_vasp_round_trip(square: rmc.AtomicStructure, tmp_path: Path) -> None:
    path = tmp_path / "POSCAR"
    rmc.write_vasp(square, BOX, path)
    data = rmc.read_vasp(path)
    np.testing.assert_allclose(data.box, BOX)
    assert data.periodic_bc().periodic
    expected, got = by_element(square), by_element(data.structure)
    assert expected.keys() == got.keys()
    for element in expected:
        np.testing.assert_allclose(got[element], expected[element], atol=1e-6)


def test_lammps_round_trip_with_legend(square: rmc.AtomicStructure, tmp_path: Path) -> None:
    path = tmp_path / "square.lammps"
    rmc.write_lammps_data(square, BOX, path)
    data = rmc.read_lammps_data(path, ["Cu", "Zr"])
    assert data.structure.elements == square.elements
    np.testing.assert_allclose(data.box, BOX)
    np.testing.assert_allclose(data.structure.coordinates, square.coordinates, atol=1e-6)
    np.testing.assert_allclose(data.origin, np.zeros(3))


def test_read_structure_by_ext_picks_the_reader(
    square: rmc.AtomicStructure, tmp_path: Path
) -> None:
    vasp = tmp_path / "cell.vasp"
    rmc.write_vasp(square, BOX, vasp)
    periodic = rmc.read_structure_by_ext(vasp)
    assert periodic.bc.periodic
    np.testing.assert_allclose(periodic.bc.box, BOX)

    pdb = tmp_path / "square.pdb"
    rmc.write_pdb(square, pdb)
    open_space = rmc.read_structure_by_ext(pdb, default_bc=rmc.InfiniteBC(volume=64.0))
    assert not open_space.bc.periodic
    assert open_space.bc.volume == 64.0


def test_read_structure_with_an_explicit_format(
    square: rmc.AtomicStructure, tmp_path: Path
) -> None:
    path = tmp_path / "no_extension"
    rmc.write_pdb(square, path)
    assert len(rmc.read_structure(path, rmc.StructFormat.Pdb).structure) == 4


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.pdb", rmc.StructFormat.Pdb),
        ("a.vasp", rmc.StructFormat.Vasp),
        ("a.POSCAR", rmc.StructFormat.Vasp),
        ("CONTCAR", rmc.StructFormat.Vasp),
        ("a.lmp", rmc.StructFormat.Lammps),
        ("a.data", rmc.StructFormat.Lammps),
        ("a.xyz", None),
    ],
)
def test_classify_structure_format(name: str, expected: rmc.StructFormat | None) -> None:
    assert rmc.classify_structure_format(Path(name)) == expected


def test_xy_and_column_data(tmp_path: Path) -> None:
    xy = tmp_path / "gr.dat"
    xy.write_text("1.0 0.5\n2.0 1.5\n3.0 1.0\n")
    np.testing.assert_allclose(rmc.read_xy_data(xy), [[1.0, 0.5], [2.0, 1.5], [3.0, 1.0]])
    table = tmp_path / "adf.dat"
    table.write_text("0 1 2 3\n10 11 12 13\n")
    np.testing.assert_allclose(rmc.read_columns(table), [[0, 1, 2, 3], [10, 11, 12, 13]])


def test_checkpoint_round_trip_loads_in_place(
    square: rmc.AtomicStructure, tmp_path: Path
) -> None:
    path = tmp_path / "run.ckpt"
    stats = rmc.EngineStats(steps_total=10, steps_accepted=4, steps_tried=9, last_total_err=1.5)
    rmc.save_checkpoint(square, stats, path)

    target = square.copy()
    view = target.coordinates
    target.coordinates = np.zeros((4, 3))
    assert rmc.load_checkpoint(target, path) == stats
    np.testing.assert_array_equal(view, square.coordinates)  # same buffer, new values


def test_checkpoint_for_another_atom_count_is_refused(
    square: rmc.AtomicStructure, tmp_path: Path
) -> None:
    path = tmp_path / "run.ckpt"
    rmc.save_checkpoint(square, rmc.EngineStats(), path)
    pair = rmc.AtomicStructure(np.zeros((2, 3)), ["Cu", "Cu"])
    with pytest.raises(rmc.IoError, match="holds 4 atoms; the structure has 2"):
        rmc.load_checkpoint(pair, path)
    np.testing.assert_array_equal(pair.coordinates, np.zeros((2, 3)))


def test_engine_stats_is_a_described_struct() -> None:
    stats = rmc.EngineStats()
    assert (stats.steps_total, stats.steps_accepted, stats.steps_tried) == (0, 0, 0)
    assert stats.last_total_err == 0.0
    assert repr(rmc.EngineStats(steps_total=3)) == (
        "EngineStats(steps_total=3, steps_accepted=0, steps_tried=0, last_total_err=0.0)"
    )
    assert rmc.EngineStats(steps_total=3) == rmc.EngineStats(steps_total=3)
    assert rmc.EngineStats(steps_total=3) != rmc.EngineStats()
    stats.steps_total = 5
    assert stats.steps_total == 5
    with pytest.raises(TypeError):
        rmc.EngineStats(3)  # type: ignore[misc]
