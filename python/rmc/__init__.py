"""Reverse Monte Carlo structural refinement.

Refines an atomic structure against experimental G(r), S(Q) and bond-angle
distribution data. The C++ engine lives in :mod:`rmc._core`; import from
:mod:`rmc`, which re-exports it with a stable ``__all__``::

    import rmc

    config = rmc.RMCConfig(pdb_path="start.pdb", pdf_path="gr.dat", rho0=0.07, steps=50_000)
    engine = rmc.refine(config, chi2_csv="chi2.csv")  # writes config.out_path
    print(engine.stats, engine.constraints.error_breakdown())

    start = rmc.make_random_amorphous(["Cu", "Zr"], [32, 32], seed=7)
    engine = rmc.Engine(start.structure, start.periodic_bc())
    engine.build_atomic_groups(0.0, 0.2)
    engine.run(10_000)                                # the GIL is released
"""

from __future__ import annotations

# The public surface is __all__ below; test_package.py checks it covers every
# public name in _core, so the star import cannot drift from it.
from ._core import *  # noqa: F403
from ._core import __version__

# The star import bound `mcsqs` to the raw _core.mcsqs submodule, and
# `from . import` returns an existing attribute rather than importing, so drop
# it first: rmc.mcsqs is the Python module over _core.mcsqs.
del mcsqs  # noqa: F821
from . import mcsqs
from . import errors
from ._protocols import (
    ConstraintProtocol,
    MoveGeneratorProtocol,
    python_constraint,
    python_generator,
)

# The validated model replaces the raw struct the star import bound: the
# builder functions still take rmc._core.RMCConfig, which RMCConfig.to_core()
# returns.
from .config import RMCConfig, refine, refine_ensemble

__all__ = sorted(
    [
        "AdfParams",
        "AdfResult",
        "Amplitude",
        "AnalysisError",
        "AngleAgitationGenerator",
        "AngleConstraint",
        "AngularDistributionConstraint",
        "AnnealingSampler",
        "AtomicStructure",
        "AtomsCollector",
        "BondConstraint",
        "BoundaryConditions",
        "Chi2CollectorCallback",
        "ClusterCorrelationConstraint",
        "ClusterOrbit",
        "ConfigError",
        "Constraint",
        "ConstraintCollection",
        "ConstraintProtocol",
        "CoordinationConstraint",
        "DihedralAngleConstraint",
        "DirectionalOrderSelector",
        "DistanceAgitationGenerator",
        "DistanceScope",
        "Engine",
        "EngineStats",
        "ExperimentalData",
        "GrParams",
        "GrResult",
        "GreedySampler",
        "Group",
        "HistogramCallback",
        "ImproperAngleConstraint",
        "InfiniteBC",
        "InterMolecularDistanceConstraint",
        "IntraMolecularDistanceConstraint",
        "IoError",
        "LabeledCurve",
        "LammpsAtomStyle",
        "LammpsData",
        "LangevinRotationGenerator",
        "LangevinTranslationGenerator",
        "LeapfrogTranslationGenerator",
        "LoadedStructure",
        "McsqsError",
        "MetropolisSampler",
        "MoveGenKind",
        "MoveGenerator",
        "MoveGeneratorCollector",
        "MoveGeneratorProtocol",
        "OrderedSelector",
        "OrientationGenerator",
        "PDBSnapshotCallback",
        "PairCorrelationConstraint",
        "PairDistributionConstraint",
        "PeriodicBC",
        "RMCConfig",
        "RandomSelector",
        "RandomStructure",
        "RandomStructureError",
        "RecursiveGroupSelector",
        "RecursiveMode",
        "ReducedStructureFactorConstraint",
        "RemoveGenerator",
        "RmcError",
        "RotationAboutAxisGenerator",
        "RotationAboutAxisPath",
        "RotationAboutSymmetryAxisGenerator",
        "RotationGenerator",
        "SmartRandomSelector",
        "SpeciesCount",
        "SpeciesSwapGenerator",
        "StructFormat",
        "StructureFactorConstraint",
        "SwapCentersGenerator",
        "SwapGenerator",
        "SymmetryAxis",
        "TranslationAlongAxisGenerator",
        "TranslationAlongAxisPath",
        "TranslationAlongSymmetryAxisGenerator",
        "TranslationGenerator",
        "TranslationTowardsAxisGenerator",
        "TranslationTowardsCentreGenerator",
        "TranslationTowardsSymmetryAxisGenerator",
        "VaspData",
        "WeightedRandomSelector",
        "__version__",
        "allocated_cpus",
        "apply_move_generator",
        "attach_constraints",
        "build_engine",
        "classify_structure_format",
        "compute_adf",
        "compute_gr",
        "default_concurrency",
        "errors",
        "has_tbb",
        "load_checkpoint",
        "load_experimental_data",
        "load_structure",
        "make_random_amorphous",
        "mcsqs",
        "periodic_box_or_zero",
        "python_constraint",
        "python_generator",
        "read_columns",
        "read_lammps_data",
        "read_pdb",
        "read_structure",
        "read_structure_by_ext",
        "read_vasp",
        "read_xy_data",
        "refine",
        "refine_ensemble",
        "run_ensemble",
        "run_ensemble_cooperative",
        "save_checkpoint",
        "set_max_concurrency",
        "write_adf",
        "write_gr",
        "write_lammps_data",
        "write_pdb",
        "write_structure_by_ext",
        "write_vasp",
    ]
)
