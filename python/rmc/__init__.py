"""Reverse Monte Carlo structural refinement.

Refines an atomic structure against experimental G(r), S(Q) and bond-angle
distribution data. The C++ engine lives in :mod:`rmc._core`; import from
:mod:`rmc`, which re-exports it with a stable ``__all__``::

    import rmc

    start = rmc.make_random_amorphous(["Cu", "Zr"], [32, 32], seed=7)
    atoms = start.structure             # AtomicStructure; atoms.coordinates is (N, 3)
    bc = start.periodic_bc()            # PeriodicBC over the generated cell
    print(atoms, bc)
"""

from __future__ import annotations

from ._core import (
    AnalysisError,
    AtomicStructure,
    AtomsCollector,
    BoundaryConditions,
    ConfigError,
    DistanceScope,
    InfiniteBC,
    IoError,
    LammpsAtomStyle,
    McsqsError,
    MoveGenKind,
    PeriodicBC,
    RandomStructure,
    RandomStructureError,
    RecursiveMode,
    RmcError,
    StructFormat,
    SymmetryAxis,
    __version__,
    allocated_cpus,
    default_concurrency,
    has_tbb,
    make_random_amorphous,
    set_max_concurrency,
)

__all__ = sorted(
    [
        "AnalysisError",
        "AtomicStructure",
        "AtomsCollector",
        "BoundaryConditions",
        "ConfigError",
        "DistanceScope",
        "InfiniteBC",
        "IoError",
        "LammpsAtomStyle",
        "McsqsError",
        "MoveGenKind",
        "PeriodicBC",
        "RandomStructure",
        "RandomStructureError",
        "RecursiveMode",
        "RmcError",
        "StructFormat",
        "SymmetryAxis",
        "__version__",
        "allocated_cpus",
        "default_concurrency",
        "has_tbb",
        "make_random_amorphous",
        "set_max_concurrency",
    ]
)
