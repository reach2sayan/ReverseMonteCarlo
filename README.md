# ReverseMonteCarlo

A complete **C++23** rewrite of the [fullrmc](https://github.com/bachiraoun/fullrmc) Reverse Monte Carlo structural refinement library.

Given experimental structural data (PDF/g(r), S(Q), neutron/X-ray scattering patterns), the engine iteratively perturbs atomic positions via a Metropolis-like acceptance criterion until the computed data matches experiment.

## Features

- **Modular architecture** — composable constraints, generators, and selectors
- **Triclinic PBC** — full periodic boundary conditions via 3×3 box matrix (`PeriodicBC`) or non-periodic systems (`InfiniteBC`)
- **10 constraint types** — Bond, Angle, Dihedral, Improper, Distance (intra/inter-molecular), Coordination, PDF g(r), Pair correlation F(r), Structure factor S(Q)
- **Move generators** — Translation (uniform, along-axis, towards-centre), Rotation (random/fixed-axis), Swap, Remove, Combined
- **Group selectors** — Random, Ordered, Weighted, Smart adaptive (ML-like weight update)
- **Boost-powered** — `boost::random` MT19937, `boost::histogram` for PDF binning, `boost::serialization` checkpoints, `boost::log` structured logging, `boost::accumulators` rolling statistics, `boost::container::flat_map` for cache-friendly lookups
- **Eigen linear algebra** — N×3 row-major coordinate matrix, Rodrigues rotations, matrix–vector S(Q) Fourier transform
- **C++23** — `std::expected` error handling, `std::span`, `std::variant`+`std::visit` zero-cost BC dispatch, concepts

## Dependencies

| Dependency | Version |
|---|---|
| CMake | ≥ 3.28 |
| C++ compiler | C++23 (GCC ≥ 13, Clang ≥ 17, MSVC 19.38+) |
| [Eigen](https://eigen.tuxfamily.org) | ≥ 3.4 |
| [Boost](https://www.boost.org) | ≥ 1.83 |
| [Catch2](https://github.com/catchorg/Catch2) | ≥ 3 (tests only) |

## Build

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build library + CLI
cmake --build build --config Release

# Run tests
ctest --test-dir build --output-on-failure
```

## CLI Usage

```bash
# Refine against a pair distribution function
./build/fullrmc_run \
    --pdb input.pdb \
    --pdf experimental_gr.dat \
    --rho0 0.033 \
    --box "20.0 20.0 20.0" \
    --steps 100000 \
    --out refined.pdb \
    --smart

# With S(Q) constraint
./build/fullrmc_run \
    --pdb input.pdb \
    --sq experimental_sq.dat \
    --rho0 0.033 \
    --steps 50000 \
    --checkpoint run.ckpt \
    --out refined.pdb
```

### Options

| Flag | Description |
|---|---|
| `--pdb` | Input structure (PDB format) |
| `--pdf` | Experimental G(r) data (two-column text) |
| `--sq` | Experimental S(Q) data (two-column text) |
| `--rho0` | Number density in atoms/Å³ |
| `--box` | Box vectors: `"a b c"` (orthogonal) or `inf` for non-periodic |
| `--steps` | Number of MC trial moves |
| `--out` | Output PDB path (default: `refined.pdb`) |
| `--checkpoint` | Save/restore binary checkpoint every 5000 accepted moves |
| `--smart` | Enable adaptive group selector (ML-like weight updates) |
| `--seed` | RNG seed for reproducibility |
| `--verbose` | Enable debug logging |

## Project Structure

```
include/fullrmc/
├── core/           Types, BoundaryConditions, Structure, Group, AtomsCollector
├── generators/     Translations, Rotations, Swaps, Removes, Combined
├── selectors/      Random, Ordered, Weighted, Smart
├── constraints/    Bond, Angle, Dihedral, Improper, Distance, Coordination,
│                   PairDistribution, PairCorrelation, StructureFactor, Collection
└── io/             PdbReader, DataReader, Checkpoint

src/                Engine.cpp, main.cpp, constraint/IO implementations
tests/              Catch2 unit + integration tests
```

## License

MIT
