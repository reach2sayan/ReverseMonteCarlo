#pragma once
#include <Eigen/Geometry>
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/core/RngGenerator.hpp>
#include <RMC/generators/GradientOracle.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/assert.hpp>

namespace RMC {

// MALA-style rotation by angle θ about a random axis through the group centroid:
// θ' = -(ε²/2)·(∂χ²/∂θ) + ε·η, η ~ N(0,1). Gradient is 2 evals. Axis re-sampled
// each call; drift steers angle magnitude/sign, not axis direction.
struct LangevinRotationGenerator
    : MoveGeneratorBase<LangevinRotationGenerator> {

  double step_size{0.01}; // ε (radians)
  ConstraintCollection *constraints{nullptr};
  mutable RngBuffer<> rng;

  LangevinRotationGenerator() = default;
  LangevinRotationGenerator(double eps, ConstraintCollection &c,
                            std::uint32_t seed = 42)
      : step_size(eps), constraints(&c), rng(seed) {}

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    BOOST_ASSERT_MSG(constraints,
                     "LangevinRotationGenerator: constraints pointer is null");

    vec3_t axis(rng.normal(), rng.normal(), rng.normal());
    const double n = axis.norm();
    axis = (n < 1e-12) ? vec3_t::UnitZ() : (axis / n).eval();
    const vec3_t pivot = centroid(coords, indices);
    const double g_theta = GradientOracle::rotation_gradient(
        coords, indices, axis, pivot, *constraints);
    const double theta =
        -(0.5 * step_size * step_size) * g_theta + step_size * rng.normal();

    Eigen::AngleAxisd rot(theta, axis);
    coords(indices, Eigen::all).rowwise() -= pivot.transpose();
    coords(indices, Eigen::all) =
        (rot * coords(indices, Eigen::all).transpose()).transpose();
    coords(indices, Eigen::all).rowwise() += pivot.transpose();
  }
};

} // namespace RMC
