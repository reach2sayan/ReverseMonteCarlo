// Catch2 microbenchmarks for engine throughput, constraint cost,
// selector overhead, and generator cost.
//
// Run with:  ./RMC_tests "[!benchmark]" --benchmark-samples 30
// Normal test pass skips these automatically (they are tagged [!benchmark]).

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/CoordinationConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/generators/Combined.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/OrderedSelector.hpp>
#include <RMC/selectors/RandomSelector.hpp>
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

} // namespace

// ─── 1. Step throughput vs system size
// ──────────────────────────────────────── Bare engine (no constraints),
// per-atom TranslationGenerator, RandomSelector. Measures how the fixed
// per-step overhead grows with N.
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

// ─── 2. Constraint cost isolation (N=512)
// ───────────────────────────────────── Isolates how much each constraint type
// adds to per-step cost. BondConstraint and AngleConstraint are
// O(bonds/angles); CoordinationConstraint and InterMolecularDistanceConstraint
// are O(N²).
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

  BENCHMARK_ADVANCED("+ CoordinationConstraint (O(N²))")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    CoordinationConstraint cc;
    for (int i = 0; i < N; ++i)
      cc.add_shell(static_cast<std::size_t>(i), "Ar", 2.5, 3.5, 1, 2);
    cc.set_elements(&engine.structure().elements);
    engine.add_constraint(std::move(cc));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };

  BENCHMARK_ADVANCED("+ InterMolecularDistanceConstraint (O(N²))")(
      Catch::Benchmark::Chronometer meter) {
    Engine engine(make_chain(N, 3.0), InfiniteBC(1e6));
    InterMolecularDistanceConstraint dc;
    dc.set_minimum_distance("Ar", "Ar", 2.0);
    dc.set_structure(&engine.structure().elements,
                     &engine.structure().molecule_ids);
    engine.add_constraint(std::move(dc));
    engine.build_atomic_groups(0.0, 0.2, 42);
    meter.measure([&] { engine.run(STEPS); });
  };
}

// ─── 3. Selector overhead (N=256)
// ───────────────────────────────────────────── Fixed N and generator; isolates
// the cost of each selector strategy.
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
}

// ─── 4. Generator cost (N=64, whole-molecule group)
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
