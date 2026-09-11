#!/usr/bin/env python3
"""Assert the shape of the built wheels, for both release.yml wheel jobs.

One extension, the pure-Python package, the typed marker and both stubs -- and
nothing vendored beyond oneTBB (Linux additionally allows Boost, which is shared
there; on Windows Boost is static and absorbed, so anything else appearing in
reverse_monte_carlo.libs/ means the static-Boost fallback silently did not fire).

Usage: check_wheels.py <wheel-or-glob> [...]
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

REQUIRED = (
    "rmc/__init__.py",
    "rmc/config.py",
    "rmc/mcsqs.py",
    "rmc/py.typed",
    "rmc/_core/__init__.pyi",
    "rmc/_core/mcsqs.pyi",
)
EXTENSION = re.compile(r"rmc/_core\.[^/]+\.(so|pyd)")
# libboost_*/libtbb* on Linux; tbb12 plus the MSVC runtime delvewheel pulls in.
VENDORED_OK = re.compile(r"/lib(boost_|tbb)|/(tbb12|msvcp140|vcruntime140|concrt140)", re.I)


def check(path: Path) -> bool:
    names = set(zipfile.ZipFile(path).namelist())
    print(f"=== {path}")
    ok = True
    for want in REQUIRED:
        if want not in names:
            print(f"::error::{path}: missing {want}")
            ok = False
    ext = [n for n in names if EXTENSION.fullmatch(n)]
    if len(ext) != 1:
        print(f"::error::{path}: expected exactly one extension, got {ext}")
        ok = False
    # Files only: the zip lists the directory entry itself as well.
    vendored = [
        n
        for n in names
        if n.startswith("reverse_monte_carlo.libs/") and not n.endswith("/")
    ]
    stray = [n for n in vendored if not VENDORED_OK.search(n)]
    if stray:
        print(f"::error::{path}: unexpected vendored libraries {stray}")
        ok = False
    print(f"    extension {ext[0] if ext else '-'}, vendored {sorted(vendored)}")
    return ok


def main(argv: list[str]) -> int:
    wheels = sorted(Path(p) for arg in argv for p in __import__("glob").glob(arg))
    if not wheels:
        print(f"::error::no wheels matched {argv}")
        return 1
    return 0 if all([check(w) for w in wheels]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
