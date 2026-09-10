#pragma once
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/generators/GradientOracle.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/assert.hpp>
#include <cmath>
#include <optional>

namespace RMC {

// HMC-style translation via leapfrog dynamics for the group atoms.
//
// Algorithm (velocity Verlet / leapfrog):
//   p ~ N(0, I_{3k}),  H₀ = χ²(r)/2 + ||p||²/2
//   p -= (ε/2)·∇χ²(r)
//   for t in 1..L:
//       r += ε·p
//       p -= ε·∇χ²(r)  [(ε/2)·∇χ²(r) on the last step]
//   H₁ = χ²(r')/2 + ||p'||²/2
//   accept with min(1, exp(H₀ − H₁))   [HMC Metropolis correction]
//
// If nuts_mode = true, the trajectory stops early at a U-turn:
// p · (r − r₀) < 0.
struct LeapfrogTranslationGenerator
    : MoveGeneratorBase<LeapfrogTranslationGenerator> {
  int n_steps{10};         // L: number of leapfrog steps
  double step_size{0.005}; // ε (Angstrom)
  bool nuts_mode{false};
  ConstraintCollection *constraints{nullptr};
  Rng rng;

  LeapfrogTranslationGenerator() = default;
  LeapfrogTranslationGenerator(int L, double eps, ConstraintCollection &c,
                               std::uint32_t seed = 42)
      : n_steps(L), step_size(eps), constraints(&c), rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    BOOST_ASSERT_MSG(constraints,
                     "LeapfrogTranslationGenerator: constraints pointer is null");
    const auto k = static_cast<Eigen::Index>(indices.size());
    const auto grad = [&] {
      return GradientOracle::translation_gradient(coords, indices, *constraints);
    };
    const auto hamiltonian = [&](const vec_t &p) {
      constraints->compute_after_move(coords, indices);
      return 0.5 * (constraints->total_error() + p.squaredNorm());
    };

    vec_t p = vec_t::NullaryExpr(3 * k, [&] { return rng.normal(); });
    const double H0 = hamiltonian(p);
    const coords_t saved = coords(indices, Eigen::all);

    p -= (0.5 * step_size) * grad();
    for (int t = 0; t < n_steps; ++t) {
      coords(indices, Eigen::all) += step_size * p.reshaped<Eigen::RowMajor>(k, 3);
      p -= (t == n_steps - 1 ? 0.5 : 1.0) * step_size * grad();
      if (nuts_mode && p.reshaped<Eigen::RowMajor>(k, 3)
                               .cwiseProduct(coords(indices, Eigen::all) - saved)
                               .sum() < 0.0) {
        break;
      }
    }

    const double H1 = hamiltonian(p);
    const bool rejected = H1 > H0 && rng.uniform() > std::exp(H0 - H1);
    if (rejected) {
      coords(indices, Eigen::all) = saved;
    }
    rejection_hint_ = rejected;
  }

  // Overrides MoveGeneratorBase::rejection_override; read by Engine::settle().
  [[nodiscard]] std::optional<bool> rejection_override() const noexcept {
    return rejection_hint_;
  }

private:
  std::optional<bool> rejection_hint_;
};

} // namespace RMC
