// Catch2 microbenchmarks for engine throughput, constraint cost,
// selector overhead, generator cost, and ensemble parallelism.
//
// Run with:  ./RMC_tests "[!benchmark]" --benchmark-samples 30

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <RMC/Engine.hpp>
#include <RMC/Ensemble.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/CoordinationConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/PairCorrelationConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <RMC/generators/Combined.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/OrderedSelector.hpp>
#include <RMC/selectors/RandomSelector.hpp>
#include <RMC/selectors/RecursiveGroupSelector.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <numeric>

using namespace RMC;

namespace {

AtomicStructure make_chain(int N, double spacing) {
  AtomicStructure s;
  s.coordinates.resize(N, 3);
  s.coordinates.setZero();
  s.atomic_numbers.resize(N);
  s.atomic_numbers.setOnes();
  for (int i = 0; i < N; ++i) {
    s.coordinates(i, 0) = i * spacing;
    s.names.push_back("Ar");
    s.elements.push_back("Ar");
    s.residues.push_back("ARG");
    s.molecule_ids.push_back(static_cast<std::size_t>(i));
  }
  return s;
}

// Build a PairDistributionConstraint ready for use with N atoms at rho0.
PairDistributionConstraint make_pdf(int N, double rho0 = 0.03) {
  PairDistributionConstraint pdc;
  const int n_bins = 100;
  mat_t exp_data(n_bins, 2);
  for (int i = 0; i < n_bins; ++i) {
    exp_data(i, 0) = 0.1 * (i + 1);
    exp_data(i, 1) = 0.0;
  }
  pdc.set_experimental_data(exp_data);
  pdc.set_number_density(rho0);
  (void)N;
  pdc.initialise();
  return pdc;
}

PairCorrelationConstraint make_pcf(int N, double rho0 = 0.03) {
  PairCorrelationConstraint pcf;
  const int n_bins = 100;
  mat_t exp_data(n_bins, 2);
  for (int i = 0; i < n_bins; ++i) {
    exp_data(i, 0) = 0.1 * (i + 1);
    exp_data(i, 1) = 0.0;
  }
  pcf.set_experimental_data(exp_data);
  pcf.set_number_density(rho0);
  (void)N;
  pcf.initialise();
  return pcf;
}

} // namespace

// 1. Step throughput vs system size
TEST_CASE("bench: step throughput vs system size", "[!benchmark]") {
  constexpr int STEPS = 500;

  BENCHMARK_ADVANCED("N=64   bare")(Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(64, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("N=256  bare")(Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(256, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("N=1024 bare")(Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(1024, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("N=4096 bare")(Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(4096, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };
}

// 2. Constraint cost isolation (N=512)
// Isolates marginal per-step cost of each
// constraint type. PDF and PCF are O(N²); coordination is O(N) via incremental
// update; bond/angle are O(bonds) via ItemCache.
TEST_CASE("bench: constraint cost isolation (N=512)", "[!benchmark]") {
  constexpr int N = 512;
  constexpr int STEPS = 500;

  BENCHMARK_ADVANCED("bare")(Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("+ BondConstraint (N-1 bonds)")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    BondConstraint bc;
    for (int i = 0; i < N - 1; ++i)
      bc.add_bond(static_cast<std::size_t>(i), static_cast<std::size_t>(i + 1),
                  2.5, 3.5);
    engine.add_constraint(std::move(bc));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("+ AngleConstraint (N-2 angles)")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    AngleConstraint ac;
    for (int i = 0; i < N - 2; ++i)
      ac.add_angle(static_cast<std::size_t>(i), static_cast<std::size_t>(i + 1),
                   static_cast<std::size_t>(i + 2), 2.0, 3.14);
    engine.add_constraint(std::move(ac));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("+ CoordinationConstraint O(N) incremental")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    CoordinationConstraint cc;
    for (int i = 0; i < N; ++i)
      cc.add_shell(static_cast<std::size_t>(i), "Ar", 2.5, 3.5, 1, 2);
    cc.set_elements(engine.structure().elements);
    engine.add_constraint(std::move(cc));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("+ PairDistributionConstraint O(N²)")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.add_constraint(make_pdf(N));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("+ PairCorrelationConstraint O(N²)")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.add_constraint(make_pcf(N));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("+ InterMolecularDistanceConstraint O(N²)")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    InterMolecularDistanceConstraint dc;
    dc.set_minimum_distance("Ar", "Ar", 2.0);
    dc.set_structure(engine.structure().elements,
                     engine.structure().molecule_ids);
    engine.add_constraint(std::move(dc));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };
}

// 3. Short-circuit benefit (N=256)
// Shows how cheap-first ordering +
// early exit saves O(N²) PDF work when a cheap constraint rejects the move
// first.
//
//   "PDF only"            — baseline: O(N²) every step
//   "Bond(pass) + PDF"    — bond never rejects; PDF runs every step
//   "Bond(fail) + PDF"    — bond always rejects; PDF is skipped every step
TEST_CASE("bench: short-circuit benefit (N=256)", "[!benchmark]") {
  constexpr int N = 256;
  constexpr int STEPS = 500;

  BENCHMARK_ADVANCED("PDF only")(Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.add_constraint(make_pdf(N));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("Bond(pass) + PDF — PDF always runs")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    BondConstraint bc;
    // Wide tolerance: bond never violates, PDF always executes.
    bc.add_bond(0, 1, 0.0, 1e6);
    engine.add_constraint(std::move(bc));
    engine.add_constraint(make_pdf(N));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("Bond(fail) + PDF — PDF skipped every step")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    BondConstraint bc;
    // Impossible tolerance: bond always violates → PDF never runs.
    bc.add_bond(0, 1, 0.0, 0.001);
    engine.add_constraint(std::move(bc));
    engine.add_constraint(make_pdf(N));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };
}

// 4. Selector overhead (N=256)
// Fixed N and generator; isolates
// per-step cost of each selector strategy.
TEST_CASE("bench: selector overhead (N=256, no constraints)", "[!benchmark]") {
  constexpr int N = 256;
  constexpr int STEPS = 500;

  BENCHMARK_ADVANCED("OrderedSelector")(Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    engine.set_selector(OrderedSelector{});
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("RandomSelector")(Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    engine.set_selector(RandomSelector{42});
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("SmartRandomSelector")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    engine.set_selector(SmartRandomSelector{1.1, 42});
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("WeightedRandomSelector (uniform)")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    std::vector<double> w(static_cast<std::size_t>(N), 1.0);
    engine.set_selector(WeightedRandomSelector{w, 42});
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("RecursiveGroupSelector/Refine(5) over Random")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    engine.set_selector(
        RecursiveGroupSelector{RandomSelector{42}, RecursiveMode::Refine, 5});
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("RecursiveGroupSelector/Explore(5) over Random")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.build_atomic_groups(0.0, 0.2, 42);
    engine.set_selector(
        RecursiveGroupSelector{RandomSelector{42}, RecursiveMode::Explore, 5});
    meter.measure([&] { engine.run(STEPS); });
  };
}

// 5. Ensemble parallelism scaling (N=256, PDF constraint)
// Compares wall-clock time of running 1, 2, 4, 8 independent replicas with
// run_ensemble vs the equivalent sequential steps on a single engine.
// Ideal parallel speedup = replica_count × (single_time / ensemble_time).
TEST_CASE("bench: ensemble parallelism scaling (N=500, PDF)", "[!benchmark][ensemble]") {
  constexpr int N = 500;
  constexpr std::uint64_t STEPS = 200;

  auto make = [&](std::size_t i) {
    Engine e(make_chain(N, 3.0), InfiniteBC(1e6));
    e.add_constraint(make_pdf(N));
    e.build_atomic_groups(0.0, 0.2, static_cast<std::uint32_t>(42 + i));
    return e;
  };

  BENCHMARK_ADVANCED("sequential ×1")(Catch::Benchmark::Chronometer meter) {
    meter.measure([&] { return RMC::run_ensemble(make, 1, STEPS); });
  };

  BENCHMARK_ADVANCED("parallel   ×2")(Catch::Benchmark::Chronometer meter) {
    meter.measure([&] { return RMC::run_ensemble(make, 2, STEPS); });
  };

  BENCHMARK_ADVANCED("parallel   ×4")(Catch::Benchmark::Chronometer meter) {
    meter.measure([&] { return RMC::run_ensemble(make, 4, STEPS); });
  };

  BENCHMARK_ADVANCED("parallel   ×8")(Catch::Benchmark::Chronometer meter) {
    meter.measure([&] { return RMC::run_ensemble(make, 8, STEPS); });
  };
}

// 6. Cooperative ensemble: sync interval sensitivity (N=256, PDF, 4 replicas)
// Measures overhead of broadcasting best state at different sync frequencies.
// Shorter sync_every = more broadcasts (higher overhead, faster convergence);
// longer = less overhead (approaches independent run_ensemble).
TEST_CASE("bench: cooperative ensemble sync interval (N=256, 4 replicas)",
          "[!benchmark][ensemble][cooperative]") {
  constexpr int N = 256;
  constexpr std::uint64_t TOTAL = 400;

  auto make = [&](std::size_t i) {
    Engine e(make_chain(N, 3.0), InfiniteBC(1e6));
    e.add_constraint(make_pdf(N));
    e.build_atomic_groups(0.0, 0.2, static_cast<std::uint32_t>(42 + i));
    return e;
  };

  BENCHMARK_ADVANCED("sync_every=25")(Catch::Benchmark::Chronometer meter) {
    meter.measure([&] {
      return RMC::run_ensemble_cooperative(make, 4, -1.0, 25, TOTAL);
    });
  };

  BENCHMARK_ADVANCED("sync_every=100")(Catch::Benchmark::Chronometer meter) {
    meter.measure([&] {
      return RMC::run_ensemble_cooperative(make, 4, -1.0, 100, TOTAL);
    });
  };

  BENCHMARK_ADVANCED("sync_every=200")(Catch::Benchmark::Chronometer meter) {
    meter.measure([&] {
      return RMC::run_ensemble_cooperative(make, 4, -1.0, 200, TOTAL);
    });
  };
}

// ─── 7. Generator cost (N=64, whole-molecule group)
// ─────────────────────────── Single group of 64 atoms; compares per-move cost
// across generator types.
TEST_CASE("bench: generator cost (N=64, single group)", "[!benchmark]") {
  constexpr int N = 64;
  constexpr int STEPS = 500;

  auto make_whole_group_engine = [&](auto gen) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.set_selector(OrderedSelector{});
    Group g;
    g.name = "whole";
    g.indices.resize(static_cast<std::size_t>(N));
    std::iota(g.indices.begin(), g.indices.end(), 0u);
    g.generator.emplace(std::move(gen));
    engine.add_group(std::move(g));
    return engine;
  };

  BENCHMARK_ADVANCED("TranslationGenerator")(
      Catch::Benchmark::Chronometer meter) {
    auto engine = make_whole_group_engine(TranslationGenerator(0.0, 0.2, 42));
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("RotationGenerator")(Catch::Benchmark::Chronometer meter) {
    auto engine = make_whole_group_engine(RotationGenerator(0.0, 0.2, 42));
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("CombinedMoveGenerator (translate + rotate)")(
      Catch::Benchmark::Chronometer meter) {
    auto engine = make_whole_group_engine(CombinedMoveGenerator{
        TranslationGenerator(0.0, 0.1, 42), RotationGenerator(0.0, 0.1, 43)});
    meter.measure([&] { engine.run(STEPS); });
  };
}

// 8. TBB arena thread scaling for accumulate_pair_histogram (N=512, PDF)
// Measures wall-clock time of the PDF constraint kernel at 1, 2, 4, and
// max available threads. Each benchmark caps the shared TBB arena via
// RMC::parallel::set_max_concurrency before running so results are comparable.
// Expected: near-linear speedup up to the number of physical cores.
#ifdef RMC_USE_TBB
#include <RMC/core/Parallel.hpp>
TEST_CASE("bench: PDF kernel TBB thread scaling (N=512)", "[!benchmark]") {
  constexpr int N       = 512;
  constexpr int STEPS   = 200;
  const int max_threads = RMC::parallel::default_concurrency();

  auto make = [&](int nthreads) {
    RMC::parallel::set_max_concurrency(nthreads);
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    engine.add_constraint(make_pdf(N));
    engine.build_atomic_groups(0.0, 0.2, 42);
    return engine;
  };

  BENCHMARK_ADVANCED("1 thread")(Catch::Benchmark::Chronometer meter) {
    auto engine = make(1);
    meter.measure([&] { engine.run(STEPS); });
    RMC::parallel::set_max_concurrency(max_threads);
  };

  BENCHMARK_ADVANCED("2 threads")(Catch::Benchmark::Chronometer meter) {
    auto engine = make(2);
    meter.measure([&] { engine.run(STEPS); });
    RMC::parallel::set_max_concurrency(max_threads);
  };

  BENCHMARK_ADVANCED("4 threads")(Catch::Benchmark::Chronometer meter) {
    auto engine = make(4);
    meter.measure([&] { engine.run(STEPS); });
    RMC::parallel::set_max_concurrency(max_threads);
  };

  BENCHMARK_ADVANCED("max threads")(Catch::Benchmark::Chronometer meter) {
    auto engine = make(max_threads);
    meter.measure([&] { engine.run(STEPS); });
  };
}
#endif

// 9. TBB vs serial full-histogram kernel (accumulate_pair_histogram).
//
// Benchmarks the raw O(N²) pair-counting kernel that is parallelised by TBB
// (when RMC_USE_TBB=ON).  Each case calls the free function directly so there
// is no engine or MC overhead — this isolates the histogram fill itself.
//
// Interpretation:
//   Serial path  — the #else branch of the RMC_USE_TBB dispatch.
//   TBB path     — std::for_each(par_unseq) over row indices via the arena.
//
// Run with:   ./RMC_tests "[!benchmark][tbb]" --benchmark-samples 30
//
// TBB is most beneficial at large N (≥512) where O(N²) work dwarfs thread
// overhead.  At small N the serial path can win due to scheduling costs.
#include <RMC/constraints/PairHistogram.hpp>
#include <numeric>

namespace {

// Build a uniform lattice of N atoms with 1-Å spacing.
coords_t make_lattice(int N) {
  coords_t c(N, 3);
  c.setZero();
  const int side = static_cast<int>(std::cbrt(static_cast<double>(N))) + 1;
  for (int i = 0; i < N; ++i) {
    const int z = i / (side * side); // integer (floor) division by design
    c(i, 0) = static_cast<double>(i % side);
    c(i, 1) = static_cast<double>((i / side) % side);
    c(i, 2) = static_cast<double>(z);
  }
  return c;
}

} // namespace

TEST_CASE("bench: accumulate_pair_histogram kernel scaling", "[!benchmark][tbb]") {
  // r range covers the full lattice extent; 200 bins.
  constexpr double R_MIN  = 0.0;
  constexpr double R_MAX  = 30.0;
  constexpr int    N_BINS = 200;

  auto run = [&](int N) {
    const coords_t     coords = make_lattice(N);
    const std::vector<uint8_t> elem_id(static_cast<std::size_t>(N), 0);
    const PairWeightTable      weights{};  // no per-pair weights
    vec_t hist(N_BINS);

    return [=]() mutable {
      hist.setZero();
      accumulate_pair_histogram(hist, coords, nullptr, elem_id, weights,
                                R_MIN, R_MAX, N_BINS);
      return hist.sum();  // prevent dead-code elimination
    };
  };

  BENCHMARK_ADVANCED("N=128")(Catch::Benchmark::Chronometer meter) {
    auto fn = run(128);
    meter.measure([&] { return fn(); });
  };

  BENCHMARK_ADVANCED("N=256")(Catch::Benchmark::Chronometer meter) {
    auto fn = run(256);
    meter.measure([&] { return fn(); });
  };

  BENCHMARK_ADVANCED("N=512")(Catch::Benchmark::Chronometer meter) {
    auto fn = run(512);
    meter.measure([&] { return fn(); });
  };

  BENCHMARK_ADVANCED("N=1024")(Catch::Benchmark::Chronometer meter) {
    auto fn = run(1024);
    meter.measure([&] { return fn(); });
  };

  BENCHMARK_ADVANCED("N=2048")(Catch::Benchmark::Chronometer meter) {
    auto fn = run(2048);
    meter.measure([&] { return fn(); });
  };
}

// Incremental (O(K·N)) vs full (O(N²)) — shows the payoff of incremental
// updates for K=1 (single-atom move) at several system sizes.
TEST_CASE("bench: incremental vs full histogram (K=1)", "[!benchmark][tbb]") {
  constexpr double R_MIN  = 0.0;
  constexpr double R_MAX  = 30.0;
  constexpr int    N_BINS = 200;

  auto setup = [&](int N) {
    coords_t coords = make_lattice(N);
    std::vector<uint8_t> elem_id(static_cast<std::size_t>(N), 0);
    PairWeightTable weights{};
    return std::make_tuple(coords, elem_id, weights);
  };

  auto bench_full = [&](int N) {
    auto [coords, elem_id, weights] = setup(N);
    vec_t hist(N_BINS);
    return [=]() mutable {
      hist.setZero();
      accumulate_pair_histogram(hist, coords, nullptr, elem_id, weights,
                                R_MIN, R_MAX, N_BINS);
      return hist.sum();
    };
  };

  auto bench_incremental = [&](int N) {
    auto [coords, elem_id, weights] = setup(N);
    const std::vector<std::size_t> moved = {0};
    vec_t hist(N_BINS);
    return [=]() mutable {
      hist.setZero();
      accumulate_moved_pairs(hist, coords, nullptr, elem_id, weights,
                             R_MIN, R_MAX, N_BINS, moved);
      return hist.sum();
    };
  };

  BENCHMARK_ADVANCED("full    N=256")(Catch::Benchmark::Chronometer meter) {
    auto fn = bench_full(256);
    meter.measure([&] { return fn(); });
  };
  BENCHMARK_ADVANCED("incr    N=256")(Catch::Benchmark::Chronometer meter) {
    auto fn = bench_incremental(256);
    meter.measure([&] { return fn(); });
  };

  BENCHMARK_ADVANCED("full    N=512")(Catch::Benchmark::Chronometer meter) {
    auto fn = bench_full(512);
    meter.measure([&] { return fn(); });
  };
  BENCHMARK_ADVANCED("incr    N=512")(Catch::Benchmark::Chronometer meter) {
    auto fn = bench_incremental(512);
    meter.measure([&] { return fn(); });
  };

  BENCHMARK_ADVANCED("full    N=1024")(Catch::Benchmark::Chronometer meter) {
    auto fn = bench_full(1024);
    meter.measure([&] { return fn(); });
  };
  BENCHMARK_ADVANCED("incr    N=1024")(Catch::Benchmark::Chronometer meter) {
    auto fn = bench_incremental(1024);
    meter.measure([&] { return fn(); });
  };
}
