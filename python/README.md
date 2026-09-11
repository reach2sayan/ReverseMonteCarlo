[![PyPI](https://img.shields.io/pypi/v/reverse-monte-carlo?logo=pypi&logoColor=white)](https://pypi.org/project/reverse-monte-carlo/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://pypi.org/project/reverse-monte-carlo/)
[![License](https://img.shields.io/badge/license-Apache--2.0-4B8BBE)](https://github.com/reach2sayan/ReverseMonteCarlo/blob/main/LICENSE)

# rmc

Reverse Monte Carlo structural refinement: move atoms at random, and keep a move
when it brings the structure's computed G(r), S(Q) or bond-angle distribution
closer to experiment. The engine is C++; `rmc` drives it from Python, with
coordinates as NumPy arrays and the refinement running with the GIL released.

```sh
pip install reverse-monte-carlo
```

The distribution is `reverse-monte-carlo` and the import is `rmc`. Wheels are for
Linux x86-64 (manylinux_2_28, CPython 3.11–3.14) and need an AVX2 CPU; Boost and
oneTBB are inside the wheel. Elsewhere, build from source (below).

## Refine from a configuration

`RMCConfig` holds the same knobs as the `RMC_run` command line, validated;
`refine` builds the engine, runs it and writes the result:

```python
import rmc

config = rmc.RMCConfig(
    pdb_path="input.pdb",               # or lammps_path / vasp_path
    pdf_path="experimental_gr.dat",     # and/or sq_path, adf_path
    rho0=0.033,
    box_override=(20.0, 20.0, 20.0),    # or "inf"
    steps=100_000,
    out_path="refined.vasp",            # the extension picks the format
)
engine = rmc.refine(config, chi2_csv="chi2.csv")
print(engine.stats, engine.constraints.error_breakdown())
```

`refine` also takes a `sampler`, a `selector` and a `step_callback`.
`rmc.refine_ensemble(config, 4)` runs four replicas in parallel and keeps the best.

## Drive the engine

```python
start = rmc.make_random_amorphous(["Cu", "Zr"], [32, 32], seed=7)
engine = rmc.Engine(start.structure, start.periodic_bc())
engine.build_atomic_groups(0.0, 0.2)            # one translation group per atom

pdf = rmc.PairDistributionConstraint()
pdf.set_experimental_data(rmc.read_xy_data("experimental_gr.dat"))
pdf.set_number_density(0.06)
pdf.set_elements(engine.structure)              # bind to engine.structure
engine.add_constraint(pdf)

engine.set_sampler(rmc.MetropolisSampler(0.01))
engine.set_track_best()
engine.run(50_000)                              # releases the GIL

best = engine.best_structure                    # a copy
curve = engine.constraints[0].concrete().computed
```

`engine.structure.coordinates` is an `(N, 3)` float64 view into the engine,
not a copy; writing into it moves atoms.

| | |
|---|---|
| Structures and files | `AtomicStructure`, `read_pdb`/`write_pdb`, `read_vasp`/`write_vasp`, `read_lammps_data`/`write_lammps_data`, `read_structure_by_ext`, `read_xy_data`, `save_checkpoint`/`load_checkpoint` |
| Boundary conditions | `PeriodicBC(box)`, `InfiniteBC(volume)` |
| Constraints | `PairDistributionConstraint`, `PairCorrelationConstraint`, `StructureFactorConstraint`, `ReducedStructureFactorConstraint`, `AngularDistributionConstraint`, `InterMolecularDistanceConstraint`, `IntraMolecularDistanceConstraint`, `CoordinationConstraint`, `BondConstraint`, `AngleConstraint`, `DihedralAngleConstraint`, `ImproperAngleConstraint`, `ClusterCorrelationConstraint` |
| Moves | `TranslationGenerator`, `RotationGenerator` and their axis variants, `OrientationGenerator`, agitations, swaps, paths, `MoveGeneratorCollector`, `SpeciesSwapGenerator`, `RemoveGenerator`, Langevin and leapfrog (gradient) moves, grouped with `Group` |
| Acceptance | `GreedySampler` (default), `MetropolisSampler`, `AnnealingSampler` |
| Group selection | `RandomSelector` (default), `WeightedRandomSelector`, `OrderedSelector`, `SmartRandomSelector`, `DirectionalOrderSelector`, `RecursiveGroupSelector` |
| Progress | `set_step_callback(fn)`, `Chi2CollectorCallback`, `PDBSnapshotCallback`, `HistogramCallback` |

Every class and function has a docstring, and the package ships type stubs.

## Constraints and moves written in Python

Any object with `compute_error(coords, moved) -> float` is a constraint, and any
object with `generate(coords, indices)` is a move:

```python
import numpy as np

class StayNearOrigin:
    cost = 0.5                                  # optional: cheapest run first
    def compute_error(self, coords: np.ndarray, moved: np.ndarray) -> float:
        return float(np.sum(coords[:, :2] ** 2))

engine.add_constraint(rmc.Constraint(StayNearOrigin()))
```

`rmc.ConstraintProtocol` and `rmc.MoveGeneratorProtocol` list the optional
hooks (`rigid`, `name`, `modifies_species`, `rejection_override`, ...). Each call
takes the GIL, so a Python constraint costs about a microsecond more per step
than a C++ one.

## Ensembles

```python
def make(i: int) -> rmc.Engine:
    engine = rmc.Engine(start.structure, start.periodic_bc())
    engine.build_atomic_groups(0.0, 0.2, seed=100 + i)
    ...                                         # constraints
    return engine

best = rmc.run_ensemble(make, 4, 20_000)        # replicas on threads, best one back
```

`run_ensemble_cooperative` shares the best structure between replicas every
`sync_every` steps.

## Analysis

```python
g = rmc.compute_gr(start.structure, start.periodic_bc(), params=rmc.GrParams(r_max=8.0))
adf = rmc.compute_adf(start.structure, start.periodic_bc())
columns = g.as_dict()                           # {"r", "total", "Cu-Cu", ...}: pandas.DataFrame(columns)
```

## Special quasirandom structures

```python
import numpy as np
from rmc import mcsqs

sqs = mcsqs.enumerate(mcsqs.parse_lattice("rndstr.in"), np.diag([2, 2, 2]), {2: 3.0})
engine = mcsqs.sqs_engine(sqs, sampler=rmc.AnnealingSampler(), seed=7)
engine.run(100_000)
rmc.write_pdb(engine.best_structure, "bestsqs.pdb")
```

## Keeping borrowed data valid

- **The engine copies the structure it is given.** Bind constraints to
  `engine.structure`, not to the structure you passed in.
- **Constraints borrow a structure's per-atom arrays.** A structure keeps its
  atom count once built; setters that would change it raise `ValueError`.
- **Gradient moves point into `engine.constraints`.** Build them after every
  constraint is added: `engine.build_langevin_groups(...)`, or
  `run_ensemble(..., prepare=...)`.

Every library failure raises a subclass of `rmc.RmcError` (`IoError`,
`AnalysisError`, `ConfigError`, `RandomStructureError`, `McsqsError`).

## Building from source

GCC 15, CMake 3.28+ and Boost 1.88+ (oneTBB and spdlog are optional to install;
spdlog is fetched when missing):

```sh
git clone https://github.com/reach2sayan/ReverseMonteCarlo && cd ReverseMonteCarlo
CXX=g++-15 pip install .
```

For development, the `python` CMake preset builds the extension in place and runs
the tests; see the [repository README](https://github.com/reach2sayan/ReverseMonteCarlo#python).
