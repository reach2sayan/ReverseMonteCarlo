// Catch2 benchmarks mirroring examples/benchmark/main.cpp (fullrmc benchmark).
// Loads data/system.pdb (20 THF molecules, 260 atoms) and
// data/experimental.gr from the committed example data directory.
//
// Run with:  ./RMC_tests "[!benchmark]" --benchmark-samples 10
//
// Two TEST_CASEs:
//   A) constraint subsets × representative group sizes (1, 5, 13, 26)
//   B) all-constraints step-count scaling, group size 13

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/ImproperAngleConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/io/DataReader.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <numbers>
#include <string>

using namespace RMC;

#ifndef THF_BENCHMARK_DATA_DIR
#define THF_BENCHMARK_DATA_DIR "examples/benchmark/data"
#endif

namespace {

static constexpr int ATOMS_PER_MOL = 13;

// Loaded once per translation unit — Catch2 re-uses the same process.
struct BenchData {
  AtomicStructure tmpl;
  mat_t gr_data;

  BenchData() {
    const std::string dir = THF_BENCHMARK_DATA_DIR;
    auto r1 = io::read_pdb((dir + "/system.pdb").c_str());
    REQUIRE(r1);
    tmpl = std::move(*r1);

    auto r2 = io::read_xy_data((dir + "/experimental.gr").c_str());
    REQUIRE(r2);
    gr_data = std::move(*r2);
  }
};

const BenchData &data() {
  static BenchData d;
  return d;
}

// ---- constraint builders (same definitions as main.cpp) -------------------

void add_bonds(BondConstraint &bc, std::size_t b) {
  bc.add_bond(b + 0, b + 1, 1.22, 1.70);
  bc.add_bond(b + 0, b + 4, 1.22, 1.70);
  bc.add_bond(b + 1, b + 2, 1.25, 1.90);
  bc.add_bond(b + 2, b + 3, 1.25, 1.90);
  bc.add_bond(b + 3, b + 4, 1.25, 1.90);
  bc.add_bond(b + 1, b + 5, 0.58, 1.22);
  bc.add_bond(b + 1, b + 6, 0.58, 1.22);
  bc.add_bond(b + 2, b + 7, 0.58, 1.22);
  bc.add_bond(b + 2, b + 8, 0.58, 1.22);
  bc.add_bond(b + 3, b + 9, 0.58, 1.22);
  bc.add_bond(b + 3, b + 10, 0.58, 1.22);
  bc.add_bond(b + 4, b + 11, 0.58, 1.22);
  bc.add_bond(b + 4, b + 12, 0.58, 1.22);
}

void add_angles(AngleConstraint &ac, std::size_t b) {
  const double lo = 95.0 * std::numbers::pi / 180.0;
  const double hi = 125.0 * std::numbers::pi / 180.0;
  const double lH = 98.0 * std::numbers::pi / 180.0;
  const double hH = 123.0 * std::numbers::pi / 180.0;
  ac.add_angle(b + 1, b + 0, b + 4, lo, hi);
  ac.add_angle(b + 0, b + 1, b + 2, lo, hi);
  ac.add_angle(b + 1, b + 2, b + 3, lo, hi);
  ac.add_angle(b + 2, b + 3, b + 4, lo, hi);
  ac.add_angle(b + 3, b + 4, b + 0, lo, hi);
  ac.add_angle(b + 5, b + 1, b + 6, lH, hH);
  ac.add_angle(b + 7, b + 2, b + 8, lH, hH);
  ac.add_angle(b + 9, b + 3, b + 10, lH, hH);
  ac.add_angle(b + 11, b + 4, b + 12, lH, hH);
  ac.add_angle(b + 5, b + 1, b + 2, lH, hH);
  ac.add_angle(b + 6, b + 1, b + 2, lH, hH);
  ac.add_angle(b + 7, b + 2, b + 1, lH, hH);
  ac.add_angle(b + 8, b + 2, b + 1, lH, hH);
  ac.add_angle(b + 7, b + 2, b + 3, lH, hH);
  ac.add_angle(b + 8, b + 2, b + 3, lH, hH);
  ac.add_angle(b + 9, b + 3, b + 2, lH, hH);
  ac.add_angle(b + 10, b + 3, b + 2, lH, hH);
  ac.add_angle(b + 9, b + 3, b + 4, lH, hH);
  ac.add_angle(b + 10, b + 3, b + 4, lH, hH);
  ac.add_angle(b + 11, b + 4, b + 3, lH, hH);
  ac.add_angle(b + 12, b + 4, b + 3, lH, hH);
}

void add_impropers(ImproperAngleConstraint &ia, std::size_t b) {
  const double lo = -15.0 * std::numbers::pi / 180.0;
  const double hi = 15.0 * std::numbers::pi / 180.0;
  ia.add_improper(b + 2, b + 0, b + 1, b + 4, lo, hi);
  ia.add_improper(b + 3, b + 0, b + 1, b + 4, lo, hi);
}

struct Flags {
  bool pdf, vdw, bond, angle, improper;
};

Engine build_engine(Flags f, int group_size) {
  const auto &d = data();
  const std::size_t N = d.tmpl.size();
  const int n_mol = static_cast<int>(N) / ATOMS_PER_MOL;
  Engine eng(d.tmpl, InfiniteBC{});

  if (f.pdf) {
    PairDistributionConstraint pdc;
    pdc.set_experimental_data(d.gr_data);
    pdc.set_number_density(static_cast<double>(N) / (20.0 * 8.0 * 20.0));
    pdc.set_elements(eng.structure().elements);
    pdc.set_molecule_ids(eng.structure().molecule_ids);
    pdc.set_exclude_intra(true);
    pdc.initialise();
    eng.add_constraint(std::move(pdc));
  }
  if (f.vdw) {
    InterMolecularDistanceConstraint dc;
    dc.set_minimum_distance("O", "O", 1.5);
    dc.set_minimum_distance("O", "C", 1.5);
    dc.set_minimum_distance("O", "H", 1.2);
    dc.set_minimum_distance("C", "C", 1.5);
    dc.set_minimum_distance("C", "H", 1.2);
    dc.set_minimum_distance("H", "H", 1.0);
    dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
    eng.add_constraint(std::move(dc));
  }
  if (f.bond) {
    BondConstraint bc;
    for (int m = 0; m < n_mol; ++m)
      add_bonds(bc, static_cast<std::size_t>(m * ATOMS_PER_MOL));
    eng.add_constraint(std::move(bc));
  }
  if (f.angle) {
    AngleConstraint ac;
    for (int m = 0; m < n_mol; ++m)
      add_angles(ac, static_cast<std::size_t>(m * ATOMS_PER_MOL));
    eng.add_constraint(std::move(ac));
  }
  if (f.improper) {
    ImproperAngleConstraint ia;
    for (int m = 0; m < n_mol; ++m)
      add_impropers(ia, static_cast<std::size_t>(m * ATOMS_PER_MOL));
    eng.add_constraint(std::move(ia));
  }

  std::uint32_t seed = 1;
  for (std::size_t base = 0; base < N;
       base += static_cast<std::size_t>(group_size)) {
    std::size_t end = std::min(base + static_cast<std::size_t>(group_size), N);
    Group g;
    g.name = "g" + std::to_string(base);
    for (std::size_t i = base; i < end; ++i)
      g.indices.push_back(i);
    g.generator.emplace(TranslationGenerator(0.0, 0.15, seed++));
    eng.add_group(std::move(g));
  }
  eng.set_selector(GroupSelector{SmartRandomSelector{1.1, 42}});
  return eng;
}

} // namespace

// ---------------------------------------------------------------------------
// A) constraint subsets × group sizes
// Group sizes sampled at 1 (per-atom), 5, 13 (one molecule), 26 (two molecules)
// ---------------------------------------------------------------------------
TEST_CASE("bench: THF constraint subsets vs group size", "[!benchmark]") {
  constexpr int STEPS = 500;

#define THF_BENCH(label, pdf, vdw, bond, angle, improper, gn)                  \
  BENCHMARK_ADVANCED(label)(Catch::Benchmark::Chronometer meter) {             \
    auto eng = build_engine({pdf, vdw, bond, angle, improper}, gn);            \
    meter.measure([&] { eng.run(STEPS); });                                    \
  }

  THF_BENCH("GN=1  none", false, false, false, false, false, 1);
  THF_BENCH("GN=1  pdf", true, false, false, false, false, 1);
  THF_BENCH("GN=1  vdw", false, true, false, false, false, 1);
  THF_BENCH("GN=1  bond", false, false, true, false, false, 1);
  THF_BENCH("GN=1  angle", false, false, false, true, false, 1);
  THF_BENCH("GN=1  improper", false, false, false, false, true, 1);
  THF_BENCH("GN=1  all", true, true, true, true, true, 1);

  THF_BENCH("GN=5  none", false, false, false, false, false, 5);
  THF_BENCH("GN=5  pdf", true, false, false, false, false, 5);
  THF_BENCH("GN=5  vdw", false, true, false, false, false, 5);
  THF_BENCH("GN=5  bond", false, false, true, false, false, 5);
  THF_BENCH("GN=5  angle", false, false, false, true, false, 5);
  THF_BENCH("GN=5  improper", false, false, false, false, true, 5);
  THF_BENCH("GN=5  all", true, true, true, true, true, 5);

  THF_BENCH("GN=13 none", false, false, false, false, false, 13);
  THF_BENCH("GN=13 pdf", true, false, false, false, false, 13);
  THF_BENCH("GN=13 vdw", false, true, false, false, false, 13);
  THF_BENCH("GN=13 bond", false, false, true, false, false, 13);
  THF_BENCH("GN=13 angle", false, false, false, true, false, 13);
  THF_BENCH("GN=13 improper", false, false, false, false, true, 13);
  THF_BENCH("GN=13 all", true, true, true, true, true, 13);

  THF_BENCH("GN=26 none", false, false, false, false, false, 26);
  THF_BENCH("GN=26 pdf", true, false, false, false, false, 26);
  THF_BENCH("GN=26 vdw", false, true, false, false, false, 26);
  THF_BENCH("GN=26 bond", false, false, true, false, false, 26);
  THF_BENCH("GN=26 angle", false, false, false, true, false, 26);
  THF_BENCH("GN=26 improper", false, false, false, false, true, 26);
  THF_BENCH("GN=26 all", true, true, true, true, true, 26);

#undef THF_BENCH
}

// ---------------------------------------------------------------------------
// B) all-constraints, group size 13 (one molecule), varying step count
// ---------------------------------------------------------------------------
TEST_CASE("bench: THF step-count scaling (all constraints, GN=13)",
          "[!benchmark]") {
  constexpr Flags ALL{true, true, true, true, true};

#define THF_STEPS(n)                                                           \
  BENCHMARK_ADVANCED(#n " steps")(Catch::Benchmark::Chronometer meter) {       \
    auto eng = build_engine(ALL, 13);                                          \
    meter.measure([&] { eng.run(n); });                                        \
  }

  THF_STEPS(5000);
  THF_STEPS(10000);
  THF_STEPS(25000);
  THF_STEPS(50000);

#undef THF_STEPS
}
