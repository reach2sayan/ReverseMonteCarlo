"""The package's shape: version, exports, and which _core was imported."""

from __future__ import annotations

import os
import re
from pathlib import Path

import rmc
from rmc import _core

REPO = Path(__file__).resolve().parents[2]


def test_version_is_cmakes() -> None:
    text = (REPO / "CMakeLists.txt").read_text()
    match = re.search(r"project\(ReverseMonteCarlo VERSION ([0-9.]+)", text)
    assert match is not None
    assert rmc.__version__ == _core.__version__ == match.group(1)


def test_all_is_sorted_and_covers_core() -> None:
    assert rmc.__all__ == sorted(rmc.__all__)
    public_core = {name for name in dir(_core) if not name.startswith("_")}
    assert public_core <= set(rmc.__all__)
    assert all(hasattr(rmc, name) for name in rmc.__all__)


def test_core_is_the_build_trees() -> None:
    """Under ctest, the extension must come from the tree just built."""
    package_dir = os.environ.get("RMC_PACKAGE_DIR")
    if package_dir is None:
        return  # an installed wheel: nothing to pin
    assert Path(_core.__file__).resolve().parent == Path(package_dir).resolve()
