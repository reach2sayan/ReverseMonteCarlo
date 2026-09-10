// ADF constraint benchmark: incremental single-atom and group moves, and the
// full-rebuild path (resync after every accept), on a periodic Cu/Zr cell.
//
// Uses only public API that already exists at 6eab89b (make_random_amorphous,
// analysis::compute_adf, the AngularDistributionConstraint setters, Engine,
// Group, TranslationGenerator, MetropolisSampler), so the same file can be
// compiled against that library for the pinned Part 7 reference comparison.
//
// Run with:  ./RMC_tests "bench: ADF constraint (N=256, periodic)"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <RMC/Engine.hpp>
#include <RMC/analysis/AngularDistribution.hpp>
#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/core/RandomStructure.hpp>
#include <RMC/generators/Translations.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace RMC;

namespace {

// Engine on a 256-atom periodic Cu/Zr cell refined against its own ADF, with
// groups of `group_size` consecutive atoms translated per step. Every move is
// accepted (T = 1e9), so each step exercises the commit path and, with
// resync_every = 1, a full histogram and grid rebuild.
Engine make_adf_engine(std::size_t group_size, unsigned resync_every) {
  const std::vector<std::string> els{"Cu", "Zr"};
  const std::vector<std::size_t> cnts{128, 128};
  auto cell = make_random_amorphous(els, cnts, 2.8, /*seed=*/5);
  const BoundaryConditions bc = cell->periodic_bc();

  analysis::AdfParams ap;
  ap.max_dis = 3.6;
  ap.n_bins = 45;
  const auto target = analysis::compute_adf(cell->structure.coordinates, bc,
                                            cell->structure.elements, ap);
  mat_t data(target->theta.size(),
             1 + static_cast<Eigen::Index>(target->partials.size()));
  data.col(0) = target->theta;
  for (std::size_t j = 0; j < target->partials.size(); ++j) {
    data.col(static_cast<Eigen::Index>(j) + 1) = target->partials[j].values;
  }

  Engine eng(cell->structure, bc);
  AngularDistributionConstraint c;
  c.set_experimental_data(data);
  c.set_cutoff(ap.max_dis);
  c.set_smoothing(ap.smooth_range);
  c.set_elements(eng.structure().elements);
  c.set_resync_interval(resync_every);
  eng.add_constraint(std::move(c));
  eng.set_sampler(MetropolisSampler{1e9});

  std::uint32_t seed = 1;
  const std::size_t N = eng.structure().size();
  for (std::size_t base = 0; base < N; base += group_size) {
    Group g;
    g.name = "g" + std::to_string(base);
    for (std::size_t i = base; i < std::min(base + group_size, N); ++i) {
      g.indices.push_back(i);
    }
    g.generator.emplace(TranslationGenerator(0.0, 0.1, seed++));
    eng.add_group(std::move(g));
  }
  return eng;
}

} // namespace

TEST_CASE("bench: ADF constraint (N=256, periodic)", "[!benchmark]") {
  constexpr std::uint64_t STEPS = 200;

  BENCHMARK_ADVANCED("incremental, GN=1")(Catch::Benchmark::Chronometer meter) {
    auto eng = make_adf_engine(1, 0);
    meter.measure([&] { eng.run(STEPS); });
  };
  BENCHMARK_ADVANCED("incremental, GN=8")(Catch::Benchmark::Chronometer meter) {
    auto eng = make_adf_engine(8, 0);
    meter.measure([&] { eng.run(STEPS); });
  };
  BENCHMARK_ADVANCED("full rebuild after each accept, GN=1")(
      Catch::Benchmark::Chronometer meter) {
    auto eng = make_adf_engine(1, 1);
    meter.measure([&] { eng.run(STEPS); });
  };
}
