#pragma once
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/generators/GradientOracle.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/assert.hpp>
#include <random>

namespace RMC {

// MALA-style translation: Δr = -(ε²/2)·∇χ²(r_group) + ε·η, η ~ N(0,I_{3k})
// The gradient is computed by central finite differences over the group atoms
// only (O(6k) constraint evaluations, k = group size). The Engine's standard
// Metropolis accept/reject applies after the move — this is an O(ε²)
// approximation to the exact MALA correction; refine with Tier 1b for large
// step sizes.
struct LangevinTranslationGenerator
    : MoveGeneratorBase<LangevinTranslationGenerator> {

  double step_size{0.01}; // ε (Angstrom)
  ConstraintCollection *constraints{nullptr};
  mutable std::mt19937 rng;

  LangevinTranslationGenerator() = default;
  LangevinTranslationGenerator(double eps, ConstraintCollection &c,
                               std::uint32_t seed = 42)
      : step_size(eps), constraints(&c), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    BOOST_ASSERT_MSG(
        constraints,
        "LangevinTranslationGenerator: constraints pointer is null");

    const auto k = static_cast<Eigen::Index>(indices.size());
    const vec_t g =
        GradientOracle::translation_gradient(coords, indices, *constraints);

    std::normal_distribution<double> nd(0.0, 1.0);
    const double half_eps_sq = 0.5 * step_size * step_size;

    for (Eigen::Index ai = 0; ai < k; ++ai) {
      const auto atom =
          static_cast<Eigen::Index>(indices[static_cast<std::size_t>(ai)]);
      for (int ax = 0; ax < 3; ++ax) {
        coords(atom, ax) += -half_eps_sq * g[3 * ai + ax] + step_size * nd(rng);
      }
    }
  }
};

} // namespace RMC
