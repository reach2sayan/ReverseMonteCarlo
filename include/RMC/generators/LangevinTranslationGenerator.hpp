#pragma once
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/core/RngGenerator.hpp>
#include <RMC/generators/GradientOracle.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/assert.hpp>
#include <ranges>

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
  mutable RngBuffer<> rng;

  LangevinTranslationGenerator() = default;
  LangevinTranslationGenerator(double eps, ConstraintCollection &c,
                               std::uint32_t seed = 42)
      : step_size(eps), constraints(&c), rng(seed) {}

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    BOOST_ASSERT_MSG(
        constraints,
        "LangevinTranslationGenerator: constraints pointer is null");

    const vec_t g =
        GradientOracle::translation_gradient(coords, indices, *constraints);

    const double half_eps_sq = 0.5 * step_size * step_size;
    for (const auto [ai, atom] : indices | std::views::enumerate) {
      const vec3_t grad = -half_eps_sq * g.segment<3>(3 * ai);
      const vec3_t noise =
          step_size * vec3_t::NullaryExpr([&] { return rng.normal(); });

      coords.row(static_cast<Eigen::Index>(atom)) += (grad + noise).transpose();
    }
  }
};

} // namespace RMC
