"""The committed stubs, and the one way they could break the package."""

from __future__ import annotations

import importlib.machinery
from pathlib import Path

import rmc


def test_the_stub_package_does_not_shadow_the_extension() -> None:
    """rmc/_core/ (stubs) sits beside the built rmc/_core.*.so. A real module
    wins over a same-named directory with no __init__.py, so the extension is
    what imports; pinned, because the other way round `import rmc` would get an
    empty namespace package and every symbol would vanish at once."""
    assert rmc._core.__file__.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
    assert rmc.__version__


def test_the_stubs_are_shipped_beside_the_module() -> None:
    package = Path(rmc.__file__).parent
    stubs = package / "_core"
    if not stubs.is_dir():
        return  # a build tree that has never had stubs generated still works
    assert (stubs / "__init__.pyi").is_file()
    assert (stubs / "mcsqs.pyi").is_file()
    assert (package / "py.typed").is_file()
