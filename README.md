# ReverseMonteCarlo

C++23 Reverse Monte Carlo structural refinement. Given experimental data (PDF g(r), S(Q)), the engine iteratively perturbs atomic positions via Metropolis acceptance until computed data matches experiment.

**Requirements:** CMake ≥ 3.28 · C++23 compiler (GCC ≥ 13, Clang ≥ 17) · Eigen ≥ 3.4 · Boost ≥ 1.83 · Catch2 ≥ 3 (tests only)

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

If Boost is not on the default path: `-DBOOST_ROOT=/opt/boost`  
To enable AddressSanitizer + UBSan: `-DENABLE_SANITIZERS=ON`

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
| `--seed` | `42` | RNG seed |
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
pdf.set_experimental("gr.dat");
pdf.set_structure(engine.structure());
pdf.set_rho0(0.033);
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

### Selectors

Controls which group is picked each step. Default is `RandomSelector`.

```cpp
#include <RMC/selectors/SmartRandomSelector.hpp>

// Adaptive: groups with higher acceptance rate are chosen more often
engine.set_selector(RMC::SmartRandomSelector(/*bias_factor*/ 1.1, /*seed*/ 42));

// Weighted: explicit per-group probabilities
engine.set_selector(RMC::WeightedRandomSelector(weights, /*seed*/ 42));
```

Available: `RandomSelector` · `OrderedSelector` · `WeightedRandomSelector` · `SmartRandomSelector` · `RecursiveGroupSelector`

### Running

```cpp
// Fixed number of steps
engine.run(100'000);

// Stop when total error drops below a threshold
engine.run_until(/*target_chi2*/ 0.01, /*max_steps*/ 1'000'000);

// Progress callback (fires every log_every steps)
engine.set_step_callback(
    [](uint64_t total, uint64_t accepted, uint64_t tried, double err) {
        std::cout << total << " steps  accepted=" << accepted << "  err=" << err << "\n";
    }, /*log_every*/ 1000);

// Periodic checkpointing
engine.set_checkpoint("run.ckpt", /*every_n_accepted*/ 5000);
```

Stats at any point:

```cpp
auto st = engine.stats();
// st.steps_total · st.steps_accepted · st.steps_tried · st.last_total_err
```

Refined coordinates: `engine.structure().coordinates` — N×3 Eigen matrix, row i = atom i.
