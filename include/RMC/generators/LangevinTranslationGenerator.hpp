#pragma once
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/generators/GradientOracle.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/assert.hpp>

namespace RMC {

// MALA-style translation: Δr = -(ε²/2)·∇χ²(r_group) + ε·η, η ~ N(0,I_{3k}).
// Gradient by central differences over the group atoms (O(6k) evals). Engine's
// Metropolis accept/reject follows; this is an O(ε²) approximation to MALA.
struct LangevinTranslationGenerator
    : MoveGeneratorBase<LangevinTranslationGenerator> {
  double step_size{0.01}; // ε (Angstrom)
  ConstraintCollection *constraints{nullptr};
  Rng rng;

  LangevinTranslationGenerator() = default;
  LangevinTranslationGenerator(double eps, ConstraintCollection &c,
                               std::uint32_t seed = 42)
      : step_size(eps), constraints(&c), rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    BOOST_ASSERT_MSG(constraints,
                     "LangevinTranslationGenerator: constraints pointer is null");
    const vec_t g =
        GradientOracle::translation_gradient(coords, indices, *constraints);
    const auto k = static_cast<Eigen::Index>(indices.size());
    coords(indices, Eigen::all) +=
        (-0.5 * step_size * step_size) * g.reshaped<Eigen::RowMajor>(k, 3) +
        step_size * coords_t::NullaryExpr(k, 3, [&] { return rng.normal(); });
  }
};

} // namespace RMC
