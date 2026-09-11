"""Reverse Monte Carlo structural refinement.

Refines an atomic structure against experimental G(r), S(Q) and bond-angle
distribution data. The C++ engine lives in :mod:`rmc._core`; import from
:mod:`rmc`, which re-exports it with a stable ``__all__``::

    import rmc

    start = rmc.make_random_amorphous(["Cu", "Zr"], [32, 32], seed=7)
    atoms = start.structure             # AtomicStructure; atoms.coordinates is (N, 3)
    bc = start.periodic_bc()            # PeriodicBC over the generated cell
    rmc.write_vasp(atoms, start.box, "POSCAR")
"""

from __future__ import annotations

# The public surface is __all__ below; test_package.py checks it covers every
# public name in _core, so the star import cannot drift from it.
from ._core import *  # noqa: F403
from ._core import __version__

__all__ = sorted(
    [
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
        "CoordinationConstraint",
        "DihedralAngleConstraint",
        "DirectionalOrderSelector",
        "DistanceAgitationGenerator",
        "DistanceScope",
        "Engine",
        "EngineStats",
        "GreedySampler",
        "Group",
        "HistogramCallback",
        "ImproperAngleConstraint",
        "InfiniteBC",
        "InterMolecularDistanceConstraint",
        "IntraMolecularDistanceConstraint",
        "IoError",
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
        "OrderedSelector",
        "OrientationGenerator",
        "PDBSnapshotCallback",
        "PairCorrelationConstraint",
        "PairDistributionConstraint",
        "PeriodicBC",
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
        "classify_structure_format",
        "default_concurrency",
        "has_tbb",
        "load_checkpoint",
        "make_random_amorphous",
        "read_columns",
        "read_lammps_data",
        "read_pdb",
        "read_structure",
        "read_structure_by_ext",
        "read_vasp",
        "read_xy_data",
        "save_checkpoint",
        "set_max_concurrency",
        "write_lammps_data",
        "write_pdb",
        "write_vasp",
    ]
)
