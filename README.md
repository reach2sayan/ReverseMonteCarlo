# ReverseMonteCarlo

[![CI](https://github.com/reach2sayan/ReverseMonteCarlo/actions/workflows/ci.yml/badge.svg)](https://github.com/reach2sayan/ReverseMonteCarlo/actions/workflows/ci.yml)

C++23 Reverse Monte Carlo structural refinement. Given experimental data (PDF g(r), S(Q)), the engine iteratively perturbs atomic positions via Metropolis acceptance until computed data matches experiment.

**Requirements:** CMake ≥ 3.28 · C++23 compiler (GCC ≥ 13, Clang ≥ 17) · Eigen ≥ 3.4 · Boost ≥ 1.83 · Intel oneAPI TBB ≥ 2021 (on by default) · Catch2 ≥ 3 (tests only)

**Optional:** Intel MKL (`ENABLE_MKL=ON`, default) · ATAT `corrdump` for the SQS extension (external tool, see [corrdump (ATAT)](#corrdump-atat) below — installed separately, not vendored)

For the architecture — the engine pipeline, the constraint / generator / sampler / selector contracts, and how to add your own — see [DESIGN.md](DESIGN.md). For copy-paste example runs end to end, see [WORKFLOW.md](WORKFLOW.md).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

If Boost is not on the default path: `-DBOOST_ROOT=/opt/boost`  
To enable AddressSanitizer + UBSan: `-DENABLE_SANITIZERS=ON`

The SQS extension (`mcsqs_rmc`) drives ATAT's `corrdump` as an external tool —
see [corrdump (ATAT)](#corrdump-atat) for how to provide it.

A minimal build (library + CLI only, no extension or examples):

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
| `RMC_BUILD_TESTS` | `ON` | Build the Catch2 test suite (`RMC_tests`). |
| `RMC_BUILD_EXAMPLES` | `ON` | Build the bundled C++ examples under `examples/`. |
| `RMC_BUILD_MCSQS` | `ON` | Build the `mcsqs_rmc` SQS-search extension. |
| `RMC_ATAT_PROVIDER` | `SYSTEM` | How `corrdump` is provided to the SQS extension: `SYSTEM` (use an installed corrdump), `FETCH` (download + build at configure time), or `SOURCE` (build from `RMC_ATAT_SOURCE_DIR`). See [corrdump (ATAT)](#corrdump-atat). |
| `ENABLE_SANITIZERS` | `OFF` | AddressSanitizer + UBSan on all targets. |

Example — release build with the kernels run serially:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRMC_USE_TBB=OFF
cmake --build build -j$(nproc)
```

### corrdump (ATAT)

The SQS extension (`mcsqs_rmc`) uses ATAT's `corrdump` to generate the cluster
basis. corrdump is run as a **separate executable over a process boundary** — it
is never linked into RMC, and ATAT is installed separately rather than bundled
with this project. ATAT is licensed [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/)
(see [NOTICE](NOTICE)). You provide corrdump yourself; `RMC_ATAT_PROVIDER`
selects how:

| `RMC_ATAT_PROVIDER` | What it does |
|---|---|
| `SYSTEM` *(default)* | Use a `corrdump` already installed on your system. Nothing is downloaded or built. Install ATAT separately: <https://axelvdw.github.io/atat/>. |
| `FETCH` | Clone ATAT at configure time into the build tree (not committed) and build only `corrdump`. Source via `RMC_ATAT_GIT_REPOSITORY` / `RMC_ATAT_GIT_TAG`. |
| `SOURCE` | Build `corrdump` from an existing ATAT checkout: `-DRMC_ATAT_SOURCE_DIR=<path>`. |

In `SYSTEM` mode the build looks for `corrdump` on your `PATH`; if found, that
path is baked in. If it isn't found, `mcsqs_rmc` still builds — supply corrdump
at runtime with `--corrdump <path>`. Resolution order at runtime is
`--corrdump` → the path baked in at build time → `corrdump` on `PATH`.

```bash
# Use an installed corrdump (on PATH) — the default
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build corrdump from a local ATAT checkout
cmake -B build -DRMC_ATAT_PROVIDER=SOURCE -DRMC_ATAT_SOURCE_DIR=$HOME/atat

# …or at runtime, point mcsqs_rmc at any corrdump
mcsqs_rmc --corrdump $HOME/bin/corrdump  ...
```

## CLI

`RMC_run` is a small multi-command driver. The default command **refines** a
structure against experimental data; mode flags (`--gr`, `--adf-compute`,
`--gen-random`) switch it into a one-shot analysis or generation tool that runs
no Monte Carlo. `--help` prints the full option schema.

```bash
./build/RMC_run \
    --pdb   input.pdb           \
    --pdf   experimental_gr.dat \
    --rho0  0.033               \
    --box   "20.0 20.0 20.0"    \
    --steps 100000              \
    --out   refined.pdb         \
    --smart
```

### Input / output

Supply **exactly one** input structure. The output format is chosen from the
`--out` extension: `.vasp`/`.poscar` (or a `POSCAR`/`CONTCAR` name) → VASP,
`.lammps`/`.lmp`/`.data` → LAMMPS data, otherwise PDB.

| Flag | Default | Description |
|---|---|---|
| `--pdb` / `-p` | — | Input structure (PDB) |
| `--lammps` / `-l` | — | Input LAMMPS data file (`atom_style atomic`); supplies its own periodic box |
| `--types` / `-t` | — | Element symbols for LAMMPS atom types, in order, e.g. `"Zr Cu Ag"` |
| `--vasp` | — | Input VASP POSCAR/CONTCAR; supplies its own cell |
| `--box` | input/`inf` | `"a b c"` for an orthogonal periodic box, or `inf`. Overrides any cell from the input file |
| `--out` / `-o` | `refined.pdb` | Output path (format inferred from extension) |

### Refinement (default command)

| Flag | Default | Description |
|---|---|---|
| `--pdf` / `-d` | — | Experimental G(r), two-column text → `PairDistributionConstraint` |
| `--sq` / `-q` | — | Experimental S(Q), two-column text → `StructureFactorConstraint` |
| `--adf` / `-a` | — | Experimental bond-angle distribution → `AngularDistributionConstraint` |
| `--rho0` | `0.1` | Number density in atoms/Å³ (used by the G(r)/S(Q) constraints) |
| `--adf-cutoff` | `3.4` | ADF bond cutoff (Å) |
| `--adf-smooth` | `2` | ADF boxcar smoothing half-width (`0` disables) |
| `--steps` / `-n` | `100000` | MC trial moves |
| `--move-gen` | `random` | Move proposer: `random` (classic walk), `langevin` (MALA) or `leapfrog` (HMC). The gradient movers steer atoms along −∇χ² toward the target |
| `--step` | `0.05` | Gradient step ε (Å) for `--move-gen langevin`/`leapfrog` |
| `--smart` | off | Adaptive selector — successful groups get picked more often |
| `--ensemble` / `-e` | `1` | Run N replicas in parallel; return the one with the lowest final χ² |
| `--checkpoint` / `-c` | — | Save/restore checkpoint every 5 000 accepted moves |
| `--seed` | `42` | RNG seed (replica i uses seed + i) |
| `--verbose` / `-v` | off | Debug logging |

Any combination of `--pdf` / `--sq` / `--adf` may be supplied; each adds its
constraint and they are refined jointly.

### Analysis & generation (one-shot, no MC)

Compute g(r) (total + per-element-pair partials) from a structure:

```bash
./build/RMC_run --pdb input.pdb --box "20 20 20" --gr \
    --rmin 0 --rmax 10 --nbins 200 --gr-out gr.dat
```

Compute the bond-angle distribution function:

```bash
./build/RMC_run --pdb input.pdb --adf-compute --adf-cutoff 3.4 --adf-out adf.dat
```

Generate a random amorphous starting structure and write it out:

```bash
./build/RMC_run --gen-random --elements "Zr Cu" --counts "50 50" \
    --spacing 3.0 --out start.pdb
```

| Flag | Default | Description |
|---|---|---|
| `--gr` | off | Compute g(r) and exit |
| `--gr-out` | `gr.dat` | Output path for g(r) |
| `--rmin` / `--rmax` | `0` / `10` | g(r) radius range (Å) |
| `--nbins` | `200` | g(r) / ADF bin count |
| `--adf-compute` | off | Compute the ADF and exit |
| `--adf-out` | `adf.dat` | Output path for the ADF |
| `--gen-random` | off | Generate a random amorphous structure → `--out`, then exit |
| `--elements` | — | Element symbols for `--gen-random`, e.g. `"Zr Cu"` |
| `--counts` | — | Atom count per element for `--gen-random`, e.g. `"50 50"` |
| `--spacing` | `3.0` | Grid spacing (Å) for `--gen-random` |

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

| Generator | Header | What it does |
|---|---|---|
| `TranslationGenerator` | `generators/Translations.hpp` | Displaces the whole group by one random vector; amplitude uniform in `[min, max]` Å, direction uniform on the sphere. The header also offers axis-constrained variants: `TranslationAlongAxisGenerator`, `TranslationTowardsCentreGenerator`, `TranslationTowardsAxisGenerator`. |
| `RotationGenerator` | `generators/Rotations.hpp` | Rotates the group about a random axis through its centroid; angle uniform in `[min, max]` rad. Variants: `RotationAboutAxisGenerator` (fixed axis, e.g. a bond), `RotationAboutSymmetryAxisGenerator` (a Cartesian axis), `OrientationGenerator` (align the group's principal axis toward a target). |
| `SwapGenerator` | `generators/Swaps.hpp` | Swaps the group's coordinates with a randomly chosen same-size candidate group (identity swap). `SwapCentersGenerator` instead translates the group onto another group's centroid. |
| `LangevinTranslationGenerator` | `generators/LangevinTranslationGenerator.hpp` | MALA-style biased translation: `Δr = −(ε²/2)·∇χ² + ε·η`. Steers atoms downhill in χ² using a finite-difference gradient over the group (O(6k) constraint evals). |
| `LeapfrogTranslationGenerator` | `generators/LeapfrogTranslationGenerator.hpp` | HMC translation via leapfrog dynamics over `L` steps with its own Metropolis accept/reject (Hamiltonian). Optional NUTS-style early U-turn termination. |
| `LangevinRotationGenerator` | `generators/LangevinRotationGenerator.hpp` | MALA-style biased rotation: drifts the rotation angle along −∂χ²/∂θ about a freshly sampled random axis (only 2 gradient evals). |
| `CombinedGenerator` | `generators/Combined.hpp` | `CombinedMoveGenerator` applies a fixed sequence of generators to the group every step; `MoveGeneratorCollector` picks **one** at random per step from a runtime pool. |
| `RemoveGenerator` | `generators/Removes.hpp` | Stages removal of the group's atoms via the shared `AtomsCollector`; the engine commits or rolls back after constraint evaluation. Coordinates are untouched — constraints skip removed atoms. |
| `SpeciesSwapGenerator` | `generators/SpeciesSwap.hpp` | Swaps the *species* of a site with another differently-typed site in the **same sublattice**; coordinates fixed. Used for SQS / alloy occupancy MC. |

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

| Constraint | Header | What it scores |
|---|---|---|
| `BondConstraint` | `constraints/BondConstraint.hpp` | Per-bond `[r_min, r_max]` length windows; penalises any pair outside its range. |
| `AngleConstraint` | `constraints/AngleConstraint.hpp` | Per-triplet bond-angle `[min, max]` windows (radians) about a vertex atom. |
| `DihedralAngleConstraint` | `constraints/DihedralAngleConstraint.hpp` | Per-quadruplet torsion-angle windows (radians). |
| `ImproperAngleConstraint` | `constraints/ImproperAngleConstraint.hpp` | Per-quadruplet improper (out-of-plane) angle windows (radians). |
| `IntraMolecularDistanceConstraint` | `constraints/DistanceConstraint.hpp` | Minimum-distance (hard-sphere) floor between atoms **within** a molecule. |
| `InterMolecularDistanceConstraint` | `constraints/DistanceConstraint.hpp` | Minimum-distance floor between atoms in **different** molecules; per-element-pair thresholds. |
| `CoordinationConstraint` | `constraints/CoordinationConstraint.hpp` | Target coordination number in a shell `[r_min, r_max]` around centre atoms, per neighbour element. |
| `PairDistributionConstraint` | `constraints/PairDistributionConstraint.hpp` | χ² of computed vs experimental `G(r) = 4πrρ₀(g(r)−1)` (radial-weighted PDF). |
| `PairCorrelationConstraint` | `constraints/PairCorrelationConstraint.hpp` | χ² of computed vs experimental `g(r)−1` (no radial prefactor). |
| `StructureFactorConstraint` | `constraints/StructureFactorConstraint.hpp` | χ² in reciprocal space against experimental `S(Q)` (PDF Fourier-transformed via an internal Gr→SQ matrix). |
| `ReducedStructureFactorConstraint` | `constraints/ReducedStructureFactorConstraint.hpp` | As above but against the reduced `F(Q) = Q(S(Q)−1)`. |
| `ClusterCorrelationConstraint` | `constraints/ClusterCorrelationConstraint.hpp` | χ² of cluster correlation functions vs disordered targets (drives the SQS search). |

#### Geometric & topological parameters

Geometric constraints (`Bond`/`Angle`/`Dihedral`/`Improper`) are built by
repeatedly calling `add_*(atom indices…, lo, hi)`; the window bounds are the only
per-entry parameters. `CoordinationConstraint::add_shell(centre, nb_elem, r_min,
r_max, target)` plus `set_elements(...)`. The distance constraints take
`set_minimum_distance(el1, el2, d)` (and `set_structure(elements, molecule_ids)`
to learn the molecular topology).

#### Pair / structure-factor parameters

These constraints share the same configuration surface:

| Method | Purpose |
|---|---|
| `set_experimental_data(mat)` | Two-column target — `[r, G(r)]` (PDF/PCF) or `[Q, S(Q)]` / `[Q, F(Q)]` (SQ). |
| `set_elements(elements)` | Per-atom element labels, for partial-pair weighting. |
| `set_number_density(rho0)` | Bulk number density ρ₀ (atoms/Å³) used in the normalisation. |
| `set_weight(el1, el2, w)` | Override the Faber-Ziman weight for one element pair (defaults to concentration-weighted). |
| `set_shape_function(fn)` | Multiply `f(r)` into the computed profile — nanoparticle envelope correction (PDF/PCF only). |
| `set_exclude_intra(bool)` | Drop intramolecular pairs from the histogram (PDF/PCF). |
| `initialise()` | Finalise after the setters above (builds the histogram / Gr↔SQ transform). Idempotent. |

The `r`/`Q` range and bin count are taken from the experimental data grid.

`PairDistributionConstraint` and `PairCorrelationConstraint` accept an optional shape function for nanoparticle PDF corrections:

```cpp
// Spherical nanoparticle envelope (diameter in Å)
pdf.set_shape_function(RMC::spherical_shape_fn(/*diameter*/ 30.0));

// Gaussian damping
pdf.set_shape_function(RMC::gaussian_shape_fn(/*sigma*/ 15.0));
```

### Selectors

A selector controls which group is picked each step, and receives accept/reject
feedback so adaptive variants can steer subsequent picks. Default is
`RandomSelector`.

```cpp
#include <RMC/selectors/SmartRandomSelector.hpp>

// Adaptive: groups with higher acceptance rate are chosen more often
engine.set_selector(RMC::SmartRandomSelector(/*bias_factor*/ 1.1, /*seed*/ 42));

// Weighted: explicit per-group probabilities
engine.set_selector(RMC::WeightedRandomSelector(weights, /*seed*/ 42));
```

| Selector | Header | Behaviour |
|---|---|---|
| `RandomSelector` | `selectors/RandomSelector.hpp` | Uniform random pick (the default). Ignores feedback. |
| `OrderedSelector` | `selectors/OrderedSelector.hpp` | Cycles deterministically through groups `0, 1, …, N-1, 0, …`. |
| `WeightedRandomSelector` | `selectors/RandomSelector.hpp` | Fixed per-group probabilities from an explicit weight vector (falls back to uniform if the weight count doesn't match the group count). |
| `SmartRandomSelector` | `selectors/SmartRandomSelector.hpp` | Adaptive: each accepted move multiplies a group's weight by `bias_factor`, each rejection divides it. Productive groups get picked more often. |
| `RecursiveGroupSelector` | `selectors/RecursiveGroupSelector.hpp` | Wraps any selector and locks onto the last group for up to `max_retries` extra steps. `Refine` mode locks after an acceptance (exploit), `Explore` mode locks after a rejection (keep trying for a good move). |
| `DirectionalOrderSelector` | `selectors/DirectionalOrderSelector.hpp` | Cycles through groups ordered by distance from a reference point (nearest- or farthest-first), e.g. for surface-inward refinement. Order is fixed at construction from the supplied centroids. |

```cpp
// Cycle deterministically through every group.
engine.set_selector(RMC::OrderedSelector{});

// Lock onto a productive group for up to 5 extra steps after each acceptance.
engine.set_selector(RMC::RecursiveGroupSelector(
    RMC::RandomSelector{/*seed*/ 42},
    RMC::RecursiveMode::Refine, /*max_retries*/ 5));

// Refine nearest-to-a-point groups first (one centroid per group, same order).
std::vector<RMC::vec3_t> centroids;
for (const auto &g : groups)
    centroids.push_back(s.coordinates(g.indices, Eigen::all)
                            .colwise().mean().transpose());
engine.set_selector(RMC::DirectionalOrderSelector{
    /*ref*/ origin, std::move(centroids), /*nearest_first*/ true});
```

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

| Sampler | Header | Rule & parameters |
|---|---|---|
| `GreedySampler` | `sampling/GreedySampler.hpp` | Zero-temperature quench: accept iff `Δerror ≤ tolerance`. `tolerance` (default `0`) is the slack allowed per move — `0` is strict downhill, RMC's historical behaviour. |
| `MetropolisSampler` | `sampling/MetropolisSampler.hpp` | Classical RMC: accept if `ΔE ≤ 0`, else with probability `exp(−ΔE/T)`. `T` is the fixed statistical temperature — fold σ into each χ² and use `T = 2` to recover the McGreevy–Pusztai `exp(−Δχ²/2)` rule. `T → 0` degenerates to greedy. |
| `AnnealingSampler` | `sampling/AnnealingSampler.hpp` | Metropolis with a geometric cooling schedule `T(step) = max(t_min, t0·cooling^(step/interval))`. Accepts many uphill moves early, fewer as it cools. |

#### Annealing schedule

`AnnealingSampler::Schedule` parameters:

| Field | Default | Meaning |
|---|---|---|
| `t0` | `1.0` | Initial temperature. |
| `cooling` | `0.9` | Geometric factor applied each interval; must be in `(0, 1)` to cool. |
| `interval` | `10000` | Steps between successive cooling updates. |
| `t_min` | `1e-6` | Temperature floor (avoids division by zero as T → 0). |

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

### Analysis: g(r) and the ADF

Beyond the constraints, the analysis layer computes a structure's pair
distribution g(r) and angular distribution function directly (the same kernels
the CLI's `--gr` / `--adf-compute` modes use):

```cpp
#include <RMC/analysis/RadialDistribution.hpp>

RMC::analysis::GrParams gp{.r_min = 0.0, .r_max = 10.0, .n_bins = 200};
auto g = *RMC::analysis::compute_gr(s.coordinates, bc, s.elements, gp);
// g.r — bin centres; g.total — total g(r); g.partials — per-element-pair g_ab(r)
RMC::analysis::write_gr(g, "gr.dat");
```

```cpp
#include <RMC/analysis/AngularDistribution.hpp>

RMC::analysis::AdfParams ap{.max_dis = 3.4, .n_bins = 200, .smooth_range = 2};
auto a = *RMC::analysis::compute_adf(s.coordinates, bc, s.elements, ap);
RMC::analysis::write_adf(a, "adf.dat");
```

Both return a `Result<…>` (a `boost::leaf::result`); dereference on success.
g(r) requires a periodic box.

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

**corrdump pipeline** — start from an ATAT `rndstr.in` primitive lattice.
`corrdump` (the vendored ATAT binary, driven via `boost::process`) enumerates
the cluster orbits, which are then expanded over the requested supercell:

```bash
mcsqs_rmc --lattice rndstr.in --supercell "2 2 2" \
          --d2 4.0 --d3 3.0 \
          --replicas 8 --steps 500000 --out bestsqs.pdb
```

This writes `bestsqs.pdb` plus an ATAT `bestsqs.out` (`str.out`) and prints the
result's correlations recomputed by `corrdump` as an independent cross-check.

**Legacy pipeline** — when `corrdump` is unavailable (or the cluster orbits were
enumerated elsewhere), supply a fixed-site PDB, a pre-enumerated cluster-orbit
file, and a binary/linear species map directly. No ATAT/`corrdump` is invoked,
so the `str.out` cross-check is skipped:

```bash
mcsqs_rmc --structure rndstr.pdb --clusters clusters.out \
          --species Cu:+1,Au:-1 --replicas 8
```

The cluster file lists one orbit per block: a header line `<target> <weight>
<n_points>` followed by one line of space-separated supercell site indices per
symmetry-equivalent instance, with blank lines or `#` comments between orbits.
Sublattices are inferred from PDB residue names (one per distinct residue).

`--lattice` and `--structure` are mutually exclusive; supply exactly one.

| Flag | Default | Description |
|---|---|---|
| `--lattice` / `-L` | — | ATAT `rndstr.in` primitive lattice — enables the `corrdump` pipeline |
| `--supercell` | `2 2 2` | Supercell for `--lattice`: `n` (cubic), `nx ny nz`, or nine ints |
| `--d2` / `--d3` / `--d4` | `0` | Max pair / triplet / quadruplet cluster diameter (`--d2` required with `--lattice`) |
| `--corrdump` | vendored | Path to a `corrdump` binary (overrides the vendored build) |
| `--structure` / `-s` | — | [legacy] Fixed-site input PDB — enables the no-`corrdump` pipeline |
| `--clusters` / `-c` | — | [legacy] Pre-enumerated cluster-orbit file |
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
