# ReverseMonteCarlo

C++23 Reverse Monte Carlo structural refinement. Given experimental data (PDF g(r), S(Q)), the engine iteratively perturbs atomic positions via Metropolis acceptance until computed data matches experiment.

Requires CMake ≥ 3.28, a C++23 compiler (GCC ≥ 13, Clang ≥ 17), Eigen ≥ 3.4, and Boost ≥ 1.83. Catch2 ≥ 3 is needed for tests only.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

If Boost is not on the default path, point CMake at it:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=/opt/boost
```

To enable AddressSanitizer + UBSan:

```bash
cmake -B build -DENABLE_SANITIZERS=ON
```

---

The CLI takes a PDB input and one or more experimental data files:

```bash
./build/RMC_run \
    --pdb input.pdb \
    --pdf experimental_gr.dat \
    --rho0 0.033 \
    --box "20.0 20.0 20.0" \
    --steps 100000 \
    --out refined.pdb \
    --smart
```

```
--pdb / -p      Input structure (PDB)
--pdf / -d      Experimental G(r), two-column text
--sq  / -q      Experimental S(Q), two-column text
--rho0          Number density in atoms/Å³ (default 0.1)
--box           "a b c" for orthogonal periodic box, or "inf" (default)
--steps / -n    Number of MC trial moves (default 100 000)
--out / -o      Output PDB (default refined.pdb)
--checkpoint    Save/restore binary checkpoint every 5 000 accepted moves
--smart         Adaptive selector — groups that accept more get picked more often
--seed          RNG seed (default 42)
--verbose / -v  Debug logging
```

---

The library API is built around four concepts: structure, boundary conditions, groups, and constraints.

**Structure** holds the N×3 coordinate matrix and atom metadata:

```cpp
#include <RMC/io/PdbReader.hpp>
auto result = RMC::io::read_pdb("input.pdb");
RMC::AtomicStructure s = *result;
```

Or construct directly:

```cpp
RMC::AtomicStructure s;
s.coordinates.resize(3, 3);
s.coordinates << 0,0,0,  1,0,0,  0,1,0;
s.elements   = {"O", "H", "H"};
s.names      = {"O1", "H1", "H2"};
s.residues   = {"WAT", "WAT", "WAT"};
s.molecule_ids = {0, 0, 0};
```

**Boundary conditions** are either periodic (triclinic) or infinite:

```cpp
// Orthogonal 20 Å box
RMC::mat3_t box = RMC::mat3_t::Identity() * 20.0;
RMC::BoundaryConditions bc = RMC::PeriodicBC(box);

// Triclinic
RMC::mat3_t box;
box << a1,a2,a3, b1,b2,b3, c1,c2,c3;
RMC::BoundaryConditions bc = RMC::PeriodicBC(box);

// No periodicity
RMC::BoundaryConditions bc = RMC::InfiniteBC(volume_Å3);
```

**Engine** ties everything together:

```cpp
#include <RMC/Engine.hpp>
RMC::Engine engine(std::move(s), bc);
```

**Groups** define which atoms move together and how:

```cpp
#include <RMC/generators/Translations.hpp>
#include <RMC/generators/Rotations.hpp>

RMC::Group g;
g.name    = "water_1";
g.indices = {0, 1, 2};               // atom indices
g.generator.emplace(RMC::TranslationGenerator(
    /*min_amp*/ 0.01, /*max_amp*/ 0.2, /*seed*/ 42));
engine.add_group(std::move(g));
```

For many identical groups (one per atom, one per molecule), `build_atomic_groups` auto-creates them:

```cpp
engine.build_atomic_groups(/*min_amp*/ 0.0, /*max_amp*/ 0.2, /*seed*/ 42);
```

Available generators: `TranslationGenerator`, `RotationGenerator`, `SwapGenerator`, `LangevinTranslationGenerator`, `LeapfrogTranslationGenerator`, `LangevinRotationGenerator`, `CombinedGenerator`, `RemoveGenerator`.

**Constraints** score each proposed move:

```cpp
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>

RMC::BondConstraint bc;
bc.add_bond(/*i*/ 0, /*j*/ 1, /*r_min*/ 0.8, /*r_max*/ 1.1);
engine.add_constraint(std::move(bc));

RMC::AngleConstraint ac;
ac.add_angle(1, 0, 2, /*min_rad*/ 1.4, /*max_rad*/ 2.1);
engine.add_constraint(std::move(ac));

// Intermolecular hard-sphere distance
RMC::InterMolecularDistanceConstraint dc;
dc.set_minimum_distance("O", "O", 2.4);
dc.set_structure(engine.structure().elements, engine.structure().molecule_ids);
engine.add_constraint(std::move(dc));

// PDF g(r) against experimental data
RMC::PairDistributionConstraint pdf;
pdf.set_experimental("gr.dat");
pdf.set_structure(engine.structure());
pdf.set_rho0(0.033);
engine.add_constraint(std::move(pdf));
```

Full constraint list: `BondConstraint`, `AngleConstraint`, `DihedralAngleConstraint`, `ImproperAngleConstraint`, `IntraMolecularDistanceConstraint`, `InterMolecularDistanceConstraint`, `CoordinationConstraint`, `PairDistributionConstraint`, `PairCorrelationConstraint`, `StructureFactorConstraint`.

**Group selector** controls which group is picked each step (default is `RandomSelector`):

```cpp
#include <RMC/selectors/SmartRandomSelector.hpp>
engine.set_selector(RMC::SmartRandomSelector(/*bias_factor*/ 1.1, /*seed*/ 42));
```

Available selectors: `RandomSelector`, `OrderedSelector`, `WeightedRandomSelector`, `SmartRandomSelector`, `RecursiveGroupSelector`.

**Running** the engine:

```cpp
engine.run(100'000);

// Or with a per-step callback for logging
engine.set_step_callback([](uint64_t total, uint64_t accepted, uint64_t tried, double err) {
    std::cout << total << " steps, " << err << " error\n";
}, /*log_every*/ 1000);

// Or stop when chi² drops below a threshold
engine.run_until(/*target_chi2*/ 0.01, /*max_steps*/ 1'000'000);
```

Checkpointing:

```cpp
engine.set_checkpoint("run.ckpt", /*every_n_accepted*/ 5000);
```

Reading stats back:

```cpp
auto st = engine.stats();
// st.steps_total, st.steps_accepted, st.steps_tried, st.last_total_err
```

The refined coordinates sit in `engine.structure().coordinates` (N×3 Eigen matrix, row i = atom i).
