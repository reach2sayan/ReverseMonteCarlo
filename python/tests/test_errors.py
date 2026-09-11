"""Library failures arrive as the right RmcError subclass, message intact."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import rmc

SUBCLASSES = [
    rmc.IoError,
    rmc.AnalysisError,
    rmc.RandomStructureError,
    rmc.ConfigError,
    rmc.McsqsError,
]


@pytest.mark.parametrize("kind", SUBCLASSES, ids=lambda k: k.__name__)
def test_hierarchy(kind: type[rmc.RmcError]) -> None:
    assert issubclass(kind, rmc.RmcError)
    assert kind.__module__ == "rmc._core"
    assert kind.__doc__


def test_base_is_an_exception() -> None:
    assert issubclass(rmc.RmcError, Exception)
    assert rmc.RmcError.__doc__


def test_error_from_libRMC_keeps_its_class_and_message(tmp_path: Path) -> None:
    """The message is produced inside the absorbed library: one module, one
    set of LEAF slots, so it survives rather than degrading to the fallback."""
    missing = tmp_path / "missing.pdb"
    with pytest.raises(rmc.IoError) as info:
        rmc.read_pdb(missing)
    assert str(info.value) == f"Cannot open PDB file: {missing}"


@pytest.mark.parametrize(
    "reader",
    [rmc.read_pdb, rmc.read_vasp, rmc.read_lammps_data, rmc.read_xy_data, rmc.read_columns],
    ids=lambda f: f.__name__,
)
def test_every_reader_raises_io_error_naming_the_file(
    reader: Callable[[Path], object], tmp_path: Path
) -> None:
    with pytest.raises(rmc.IoError, match=r"missing\.txt"):
        reader(tmp_path / "missing.txt")


def test_parse_failures_are_io_errors(tmp_path: Path) -> None:
    ragged = tmp_path / "ragged.dat"
    ragged.write_text("1 2 3\n4 5\n")
    with pytest.raises(rmc.IoError, match="ragged rows"):
        rmc.read_columns(ragged)


def test_the_base_catches_them_all(tmp_path: Path) -> None:
    with pytest.raises(rmc.RmcError):
        rmc.read_pdb(tmp_path / "missing.pdb")
