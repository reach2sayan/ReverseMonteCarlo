#pragma once
#include <Eigen/Core>
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/generators/GradientOracle.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/assert.hpp>
#include <cmath>
#include <optional>
#include <random>

namespace RMC {

// HMC-style translation via leapfrog dynamics for the group atoms.
//
// Algorithm (velocity Verlet / leapfrog):
//   p ~ N(0, I_{3k}),  H₀ = χ²(r)/2 + ||p||²/2
//   g = ∇χ²(r)
//   p -= (ε/2)·g
//   for t in 1..L:
//       r += ε·p             (position full-step)
//       g  = ∇χ²(r)          (gradient at new position)
//       p -= ε·g  [or (ε/2)·g on the last step]
//   H₁ = χ²(r')/2 + ||p'||²/2
//   accept with min(1, exp(H₀ − H₁))   [HMC Metropolis correction]
//
// If nuts_mode = true, trajectory is terminated early when a U-turn is
// detected: p · (centroid(r) − centroid(r₀)) < 0.
//
struct LeapfrogTranslationGenerator
    : MoveGeneratorBase<LeapfrogTranslationGenerator> {

  int n_steps{10};         // L: number of leapfrog steps
  double step_size{0.005}; // ε (Angstrom)
  bool nuts_mode{false};
  ConstraintCollection *constraints{nullptr};
  mutable std::mt19937 rng;
  mutable std::optional<bool> rejection_hint_{std::nullopt};

  LeapfrogTranslationGenerator() = default;
  LeapfrogTranslationGenerator(int L, double eps, ConstraintCollection &c,
                               std::uint32_t seed = 42)
      : n_steps(L), step_size(eps), constraints(&c), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    BOOST_ASSERT_MSG(
        constraints,
        "LeapfrogTranslationGenerator: constraints pointer is null");

    rejection_hint_.reset();
    const auto k = static_cast<Eigen::Index>(indices.size());

    // H₀: get current χ² at pre-leapfrog position
    constraints->compute_after_move(coords, indices);
    const double chi2_0 = constraints->total_error();

    // Sample momentum
    std::normal_distribution<double> nd(0.0, 1.0);
    vec_t p = Eigen::VectorXd::NullaryExpr(3 * k, [&] { return nd(rng); });
    const double K0 = p.squaredNorm() / 2.0;
    const double H0 = chi2_0 / 2.0 + K0;

    // Save initial centroid for NUTS U-turn check (used only in nuts_mode)
    [[maybe_unused]] const vec3_t r0_centroid = centroid(coords, indices);

    // Save initial coords for rollback
    Eigen::Map<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> idx(
        indices.data(), k);
    Eigen::MatrixXd saved = coords(idx, Eigen::all);

    // Initial half-step for momentum
    vec_t g =
        GradientOracle::translation_gradient(coords, indices, *constraints);
    p -= (step_size / 2.0) * g;

    int steps_done = 0;
    for (int t = 0; t < n_steps; ++t) {
      for (Eigen::Index ai = 0; ai < k; ++ai) {
        const auto atom =
            static_cast<Eigen::Index>(indices[static_cast<std::size_t>(ai)]);
        coords.row(atom).transpose() += step_size * p.segment<3>(3 * ai);
      }

      g = GradientOracle::translation_gradient(coords, indices, *constraints);
      // Momentum update: full-step inside trajectory, half-step at end
      if (t == n_steps - 1) {
        p -= (step_size / 2.0) * g;
      } else {
        p -= step_size * g;
      }

      ++steps_done;
      // NUTS U-turn check
      if (nuts_mode) {
        vec_t r_diff_flat(3 * k);
        for (Eigen::Index ai = 0; ai < k; ++ai) {
          const auto atom =
              static_cast<Eigen::Index>(indices[static_cast<std::size_t>(ai)]);
          r_diff_flat.segment<3>(3 * ai) =
              coords.row(atom).transpose() - saved.row(ai).transpose();
        }
        if (p.dot(r_diff_flat) < 0.0)
          break;
      }
    }
    (void)steps_done;

    // H₁: χ² at leapfrog endpoint (last gradient call already set err_after_,
    // but we need an explicit evaluation for total_error() to reflect this)
    constraints->compute_after_move(coords, indices);
    const double chi2_1 = constraints->total_error();
    const double K1 = p.squaredNorm() / 2.0;
    const double H1 = chi2_1 / 2.0 + K1;

    // HMC Metropolis accept/reject
    const double log_alpha = H0 - H1;
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const bool rejected = (log_alpha < 0.0) && (ud(rng) > std::exp(log_alpha));

    if (rejected) {
      // Restore original positions; Engine's score_after will then see
      // err_after ≈ err_before and accept the no-op correctly.
      coords(idx, Eigen::all) = saved;
    }
    rejection_hint_ = rejected;
  }

  // CMoveGeneratorWithRejectionOverride — queried by Engine::settle()
  [[nodiscard]] std::optional<bool> rejection_override() const noexcept {
    return rejection_hint_;
  }
};

} // namespace RMC
