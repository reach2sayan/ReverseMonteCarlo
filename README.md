# ReverseMonteCarlo

C++23 Reverse Monte Carlo structural refinement. Given experimental data (PDF g(r), S(Q)), the engine iteratively perturbs atomic positions via Metropolis acceptance until computed data matches experiment.

**Requirements:** CMake ≥ 3.28 · C++23 compiler (GCC ≥ 13, Clang ≥ 17) · Eigen ≥ 3.4 · Boost ≥ 1.83 · Intel oneAPI TBB ≥ 2021 (on by default) · Catch2 ≥ 3 (tests only)

**Optional:** Intel MKL (`ENABLE_MKL=ON`, default) · vendored ATAT `corrdump` for the SQS extension (`RMC_BUILD_ATAT=ON`, default; pulled in as a git submodule)

## Build

The ATAT submodule backs the SQS extension — clone recursively (or fetch it after the fact):

```bash
git clone --recursive <repo-url>
# already cloned? →  git submodule update --init --recursive
```

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

If Boost is not on the default path: `-DBOOST_ROOT=/opt/boost`  
To enable AddressSanitizer + UBSan: `-DENABLE_SANITIZERS=ON`

A minimal build (library + CLI only, no extension, examples or ATAT):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DRMC_BUILD_MCSQS=OFF -DRMC_BUILD_EXAMPLES=OFF
```

### CMake options

| Option | Default | Description |
|---|---|---|
| `RMC_USE_TBB` | `ON` | Parallelise the O(N²) pair-histogram build with Intel TBB (`std::execution::par_unseq`) inside a shared task arena. Requires oneAPI TBB ≥ 2021. Set `OFF` to run the kernels serially. |
| `ENABLE_MKL` | `ON` | Use Intel MKL as the Eigen BLAS/LAPACK backend. Falls back silently if MKL is not found. |
| `ENABLE_NATIVE_ARCH` | `ON` | Compile with `-march=native` (AVX2 etc.). Set `OFF` for a portable `-march=x86-64-v3` build. |
| `ENABLE_LTO` | `ON` | Link-time optimisation (IPO), when the toolchain supports it. |
| `ENABLE_CCACHE` | `ON` | Use `ccache` as the compiler launcher when available. |
| `ENABLE_PGO_GENERATE` / `ENABLE_PGO_USE` | `OFF` | Profile-guided optimisation generate/use passes (mutually exclusive). |
| `RMC_BUILD_TESTS` | `ON` | Build the Catch2 test suite (`RMC_tests`). |
| `RMC_BUILD_EXAMPLES` | `ON` | Build the fullrmc-equivalent C++ examples under `examples/`. |
| `RMC_BUILD_MCSQS` | `ON` | Build the `mcsqs_rmc` SQS-search extension. |
| `RMC_BUILD_ATAT` | `ON` | Vendor and build ATAT's `corrdump` (driven at runtime via `boost::process`, never linked). Requires the `extern/atat` submodule. |
| `ENABLE_SANITIZERS` | `OFF` | AddressSanitizer + UBSan on all targets. |

Example — release build with the kernels run serially:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRMC_USE_TBB=OFF
cmake --build build -j$(nproc)
```

## CLI

```bash
./build/RMC_run \
    --pdb   input.pdb          \
    --pdf   experimental_gr.dat \
    --rho0  0.033              \
    --box   "20.0 20.0 20.0"  \
    --steps 100000             \
    --out   refined.pdb        \
    --smart
```

| Flag | Default | Description |
|---|---|---|
| `--pdb` / `-p` | — | Input structure (PDB) |
| `--pdf` / `-d` | — | Experimental G(r), two-column text |
| `--sq` / `-q` | — | Experimental S(Q), two-column text |
| `--rho0` | `0.1` | Number density in atoms/Å³ |
| `--box` | `inf` | `"a b c"` for orthogonal periodic box, or `inf` |
| `--steps` / `-n` | `100000` | MC trial moves |
| `--out` / `-o` | `refined.pdb` | Output PDB |
| `--checkpoint` / `-c` | — | Save/restore checkpoint every 5 000 accepted moves |
| `--smart` | off | Adaptive selector — successful groups get picked more often |
| `--ensemble` / `-e` | `1` | Run N replicas in parallel; return the one with the lowest final χ² |
| `--seed` | `42` | RNG seed (replica i uses seed + i) |
| `--verbose` / `-v` | off | Debug logging |

## Library API

### Structure

Load from PDB:

```cpp
#include <RMC/io/PdbReader.hpp>

auto result = RMC::io::read_pdb("input.pdb");
RMC::AtomicStructure s = *result;
```

Or construct directly (coordinates are an N×3 Eigen matrix, row i = atom i):

```cpp
RMC::AtomicStructure s;
s.coordinates.resize(3, 3);
s.coordinates << 0,0,0,  1,0,0,  0,1,0;
s.elements     = {"O", "H", "H"};
s.names        = {"O1", "H1", "H2"};
s.residues     = {"WAT", "WAT", "WAT"};
s.molecule_ids = {0, 0, 0};
```

### Boundary conditions

```cpp
#include <RMC/core/BoundaryConditions.hpp>

// Orthogonal box
RMC::mat3_t box = RMC::mat3_t::Identity() * 20.0;
RMC::BoundaryConditions bc = RMC::PeriodicBC(box);

// Triclinic box
RMC::mat3_t box;
box << a1,a2,a3,
       b1,b2,b3,
       c1,c2,c3;
RMC::BoundaryConditions bc = RMC::PeriodicBC(box);

// No periodicity
RMC::BoundaryConditions bc = RMC::InfiniteBC(volume_Å3);
```

### Engine

```cpp
#include <RMC/Engine.hpp>

RMC::Engine engine(std::move(s), bc);
```

### Groups

A group is a set of atom indices that move together. Attach a generator to control the move type:

```cpp
#include <RMC/generators/Translations.hpp>
#include <RMC/generators/Rotations.hpp>

RMC::Group g;
g.name    = "water_1";
g.indices = {0, 1, 2};
g.generator.emplace(RMC::TranslationGenerator(/*min*/ 0.01, /*max*/ 0.2, /*seed*/ 42));
engine.add_group(std::move(g));
```

To auto-create one group per atom with a `TranslationGenerator`:

```cpp
engine.build_atomic_groups(/*min*/ 0.0, /*max*/ 0.2, /*seed*/ 42);
```

| Generator | Header |
|---|---|
| `TranslationGenerator` | `generators/Translations.hpp` |
| `RotationGenerator` | `generators/Rotations.hpp` |
| `SwapGenerator` | `generators/Swaps.hpp` |
| `LangevinTranslationGenerator` | `generators/LangevinTranslationGenerator.hpp` |
| `LeapfrogTranslationGenerator` | `generators/LeapfrogTranslationGenerator.hpp` |
| `LangevinRotationGenerator` | `generators/LangevinRotationGenerator.hpp` |
| `CombinedGenerator` | `generators/Combined.hpp` |
| `RemoveGenerator` | `generators/Removes.hpp` |
| `SpeciesSwapGenerator` | `generators/SpeciesSwap.hpp` |

### Constraints

Each constraint scores a proposed move; the engine accepts or rejects based on the aggregate error change.

```cpp
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>

// Bond length [r_min, r_max] in Å
RMC::BondConstraint bc;
bc.add_bond(0, 1, /*r_min*/ 0.8, /*r_max*/ 1.1);
engine.add_constraint(std::move(bc));

// Bond angle [min, max] in radians
RMC::AngleConstraint ac;
ac.add_angle(1, 0, 2, /*min*/ 1.4, /*max*/ 2.1);
engine.add_constraint(std::move(ac));

// Intermolecular hard-sphere distance
RMC::InterMolecularDistanceConstraint dc;
dc.set_minimum_distance("O", "O", 2.4);
dc.set_structure(engine.structure().elements, engine.structure().molecule_ids);
engine.add_constraint(std::move(dc));

// Pair distribution function G(r) vs experiment
RMC::PairDistributionConstraint pdf;
// exp_data: Eigen matrix with columns [r, G(r)]
pdf.set_experimental_data(exp_data);
pdf.set_elements(engine.structure().elements);
pdf.set_number_density(0.033);
pdf.initialise();
engine.add_constraint(std::move(pdf));
```

| Constraint | Header |
|---|---|
| `BondConstraint` | `constraints/BondConstraint.hpp` |
| `AngleConstraint` | `constraints/AngleConstraint.hpp` |
| `DihedralAngleConstraint` | `constraints/DihedralAngleConstraint.hpp` |
| `ImproperAngleConstraint` | `constraints/ImproperAngleConstraint.hpp` |
| `IntraMolecularDistanceConstraint` | `constraints/DistanceConstraint.hpp` |
| `InterMolecularDistanceConstraint` | `constraints/DistanceConstraint.hpp` |
| `CoordinationConstraint` | `constraints/CoordinationConstraint.hpp` |
| `PairDistributionConstraint` | `constraints/PairDistributionConstraint.hpp` |
| `PairCorrelationConstraint` | `constraints/PairCorrelationConstraint.hpp` |
| `StructureFactorConstraint` | `constraints/StructureFactorConstraint.hpp` |
| `ReducedStructureFactorConstraint` | `constraints/ReducedStructureFactorConstraint.hpp` |
| `ClusterCorrelationConstraint` | `constraints/ClusterCorrelationConstraint.hpp` |

`PairDistributionConstraint` and `PairCorrelationConstraint` accept an optional shape function for nanoparticle PDF corrections:

```cpp
// Spherical nanoparticle envelope (diameter in Å)
pdf.set_shape_function(RMC::spherical_shape_fn(/*diameter*/ 30.0));

// Gaussian damping
pdf.set_shape_function(RMC::gaussian_shape_fn(/*sigma*/ 15.0));
```

### Selectors

Controls which group is picked each step. Default is `RandomSelector`.

```cpp
#include <RMC/selectors/SmartRandomSelector.hpp>

// Adaptive: groups with higher acceptance rate are chosen more often
engine.set_selector(RMC::SmartRandomSelector(/*bias_factor*/ 1.1, /*seed*/ 42));

// Weighted: explicit per-group probabilities
engine.set_selector(RMC::WeightedRandomSelector(weights, /*seed*/ 42));
```

Available: `RandomSelector` · `OrderedSelector` · `WeightedRandomSelector` · `SmartRandomSelector` · `RecursiveGroupSelector` · `DirectionalOrderSelector`

### Running

```cpp
// Fixed number of steps
engine.run(100'000);

// Stop when total error drops below a threshold
engine.run_until(/*target_chi2*/ 0.01, /*max_steps*/ 1'000'000);

// Progress callback (fires every log_every steps). The final argument is the
// current (live) structure.
engine.set_step_callback(
    [](uint64_t total, uint64_t accepted, uint64_t tried, double err,
       const RMC::AtomicStructure &cur) {
        std::cout << total << " steps  accepted=" << accepted << "  err=" << err << "\n";
    }, /*log_every*/ 1000);

// Periodic checkpointing
engine.set_checkpoint("run.ckpt", /*every_n_accepted*/ 5000);
```

### Acceptance policy (samplers)

Each trial move is accepted in three tiers: gradient-based generators (HMC /
leapfrog) own their own accept/reject; otherwise any worsened *rigid* constraint
is a hard rejection; otherwise a pluggable **sampler** decides from the change in
the soft total error. The default is `GreedySampler` — strict downhill, RMC's
historical behaviour.

```cpp
#include <RMC/sampling/MetropolisSampler.hpp>
#include <RMC/sampling/AnnealingSampler.hpp>

// Fixed-temperature Metropolis: accept ΔE ≤ 0, else with prob exp(-ΔE/T).
engine.set_sampler(RMC::MetropolisSampler{/*T*/ 2.0}, /*seed*/ 42);

// Simulated annealing: geometric cooling T(step) = max(t_min, t0·cooling^(step/interval)).
engine.set_sampler(
    RMC::AnnealingSampler{{.t0 = 1.0, .cooling = 0.9, .interval = 10'000}},
    /*seed*/ 42);

// Greedy quench with a tolerance (accept if it doesn't raise total error by > tol).
engine.set_sampler(RMC::GreedySampler{/*tolerance*/ 0.0});
```

`seed` seeds the engine's dedicated acceptance RNG; give each ensemble replica a
distinct seed for independent stochastic streams.

| Sampler | Header |
|---|---|
| `GreedySampler` | `sampling/GreedySampler.hpp` |
| `MetropolisSampler` | `sampling/MetropolisSampler.hpp` |
| `AnnealingSampler` | `sampling/AnnealingSampler.hpp` |

With annealing the *final* MC state is deliberately not the minimum, so track the
best-ever configuration:

```cpp
engine.set_track_best(true);          // off by default (costs an O(N) copy per improvement)
engine.run(500'000);
double best = engine.best_error();
const RMC::AtomicStructure &s = engine.best_structure();
```

Stats at any point:

```cpp
auto st = engine.stats();
// st.steps_total · st.steps_accepted · st.steps_tried · st.last_total_err
```

Refined coordinates: `engine.structure().coordinates` — N×3 Eigen matrix, row i = atom i.

### Ensemble runs

`#include <RMC/Ensemble.hpp>`

**Independent multi-start** — run N replicas in parallel, return the best:

```cpp
auto make_engine = [&](std::size_t replica) -> RMC::Engine {
    RMC::Engine e(structure_copy, bc);
    e.build_atomic_groups(0.0, 0.2, /*seed*/ 42 + static_cast<uint32_t>(replica));
    e.add_constraint(pdf_constraint);
    return e;
};

// Runs 4 replicas on 4 threads; returns lowest-χ² engine.
RMC::Engine best = RMC::run_ensemble(make_engine, /*n_replicas*/ 4, /*steps*/ 100'000);
```

**Cooperative (island model)** — replicas periodically share the best state:

```cpp
// Every 1 000 steps all replicas synchronise: the best engine is broadcast
// to laggards. Stops as soon as any replica reaches target_chi2.
RMC::Engine best = RMC::run_ensemble_cooperative(
    make_engine,
    /*n_replicas*/  4,
    /*target_chi2*/ 0.05,
    /*sync_every*/  1000,
    /*max_steps*/   1'000'000);
```

Both functions require that `make_engine(i)` constructs a fully configured, ready-to-run `Engine` for replica `i`. All engines are constructed in the calling thread (required for correct `boost::context` fibre lifetimes); background threads only call `.run()`.

#### TBB + ensemble thread budgeting

When `RMC_USE_TBB=ON`, each replica thread also uses TBB's thread pool for the
O(N²) histogram kernel. To prevent oversubscription, both ensemble functions
automatically cap TBB's global thread count to `allocated_cpus / n_replicas`,
where `allocated_cpus` is read from the scheduler environment in priority order:

1. `SLURM_CPUS_PER_TASK` (SLURM)
2. `PBS_NUM_PPN` (PBS/Torque)
3. `LSB_DJOB_NUMPROC` (LSF)
4. `std::thread::hardware_concurrency()` (local fallback)

For schedulers that don't export these variables, pass the budget explicitly:

```cpp
// 4 replicas, 2 TBB workers each on a 32-core allocation
RMC::run_ensemble_cooperative(make_engine, 4, 0.05, 1000, 0,
                              /*tbb_threads_per_replica*/ 8);
```

### Multi-frame refinement

`Engine` refines one or more structural frames against a single experimental
dataset. With a single frame this is ordinary RMC; add extra frames with
`add_frame()` and the chi² is evaluated on the *average* computed profile across
all frames, helping prevent over-fitting to a single configuration. (The
averaging lives inside the pair/angle constraints; single-frame is just the
N = 1 case.)

```cpp
#include <RMC/Engine.hpp>

RMC::Engine eng(trajectory[0], bc);   // frame 0

// Add the remaining frames (typically from an MD trajectory)
for (auto &frame : trajectory | std::views::drop(1))
    eng.add_frame(frame);

// Groups and constraints are shared across all frames
eng.add_group(std::move(g));

RMC::PairDistributionConstraint pdf;
// ... configure pdf ...
eng.add_constraint(std::move(pdf));

eng.initialise();   // pre-populates per-frame histograms
eng.run(100'000);
```

Each MC step picks one frame and one group at random, proposes a move, and
accepts or rejects based on whether the *averaged* chi² improves. Per-step
histogram updates are O(K·N) incremental (K = atoms in the moved group) rather
than O(N²), so cost per step is independent of N for K ≪ N.

### Pair histogram performance

| Scenario | Cost per step |
|---|---|
| Single-frame, serial | O(N²) full rebuild every step |
| Single-frame, `RMC_USE_TBB=ON` | O(N²) full rebuild, parallelised over rows |
| Multi-frame (any backend) | O(K·N) incremental update (K = group size) |

The TBB backend is most beneficial for single-engine use with large N (≥ 512).
For ensemble runs the replica-level parallelism is the primary speedup source;
TBB provides a secondary boost within each replica subject to the thread budget
described above.

## SQS search (`mcsqs_rmc`)

`mcsqs_rmc` is a separate executable (built when `RMC_BUILD_MCSQS=ON`) that
generates Special Quasirandom Structures by driving the RMC engine instead of
ATAT's `mcsqs` MC loop. It keeps the lattice sites fixed and only swaps site
occupancies (`SpeciesSwapGenerator`), scoring each configuration's cluster
correlations against their disordered targets (`ClusterCorrelationConstraint`).
Over ATAT's `mcsqs` it adds a pluggable acceptance policy (the default
`AnnealingSampler` mirrors `mcsqs`'s simulated annealing), adaptive site
selection (`SmartRandomSelector`), and optional island-model ensemble
parallelism.

**ATAT pipeline** — start from an ATAT `rndstr.in` primitive lattice. `corrdump`
(the vendored ATAT binary, driven via `boost::process`) enumerates the cluster
orbits, which are then expanded over the requested supercell:

```bash
mcsqs_rmc --lattice rndstr.in --supercell "2 2 2" \
          --d2 4.0 --d3 3.0 \
          --replicas 8 --steps 500000 --out bestsqs.pdb
```

This writes `bestsqs.pdb` plus an ATAT `bestsqs.out` (`str.out`) and prints the
result's correlations recomputed by `corrdump` as an independent cross-check.

**Legacy pipeline** — supply a fixed-site PDB, a cluster-orbit file, and a
species map directly (no ATAT/`corrdump` needed):

```bash
mcsqs_rmc --structure rndstr.pdb --clusters clusters.out \
          --species Cu:+1,Au:-1 --replicas 8
```

| Flag | Default | Description |
|---|---|---|
| `--lattice` / `-L` | — | ATAT `rndstr.in` primitive lattice — enables the `corrdump` pipeline |
| `--supercell` | `2 2 2` | Supercell for `--lattice`: `n` (cubic), `nx ny nz`, or nine ints |
| `--d2` / `--d3` / `--d4` | `0` | Max pair / triplet / quadruplet cluster diameter (`--d2` required with `--lattice`) |
| `--corrdump` | vendored | Path to a `corrdump` binary (overrides the vendored build) |
| `--structure` / `-s` | — | [legacy] Fixed-site input PDB |
| `--clusters` / `-c` | — | [legacy] Cluster-orbit file |
| `--species` / `-S` | — | [legacy] Species occupation map, e.g. `Cu:+1,Au:-1` |
| `--steps` / `-n` | `500000` | MC steps per replica |
| `--replicas` / `-r` | `1` | Island-model replicas (> 1 enables the ensemble) |
| `--sampler` | `anneal` | Acceptance policy: `greedy` \| `metropolis` \| `anneal` |
| `--T0` | `1.0` | Temperature (metropolis: fixed T; anneal: initial T) |
| `--cooling` | `0.9` | Annealing geometric cooling factor in (0,1) |
| `--cool-interval` | `0` | Steps between cooling updates (0 → `steps/20`) |
| `--seed` | `42` | RNG seed |
| `--out` / `-o` | `bestsqs.pdb` | Output SQS PDB |
| `--log-every` / `-l` | `10000` | Print progress every N steps |

Sublattices are inferred from PDB residue names (one sublattice per distinct
residue); swaps only ever exchange occupancy within a sublattice.
