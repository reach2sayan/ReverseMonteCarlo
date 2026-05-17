#pragma once
#include <Eigen/Geometry>
#include <fullrmc/generators/MoveGenerator.hpp>
#include <numbers>
#include <random>

namespace fullrmc {

// Rotates all atoms in a group about a random axis through the group centroid.
// Angle is drawn uniformly from [min_angle, max_angle] (radians).
struct RotationGenerator : MoveGeneratorBase<RotationGenerator> {
  double min_angle{0.0};
  double max_angle{0.1}; // ~5.7 degrees
  mutable std::mt19937 rng;

  RotationGenerator() = default;
  RotationGenerator(double mn, double mx, std::uint32_t seed = 42)
      : min_angle(mn), max_angle(mx), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double angle =
        (min_angle < max_angle)
            ? std::uniform_real_distribution<double>(min_angle, max_angle)(rng)
            : min_angle;
    std::uniform_real_distribution<double> sign_dist(-1.0, 1.0);
    if (sign_dist(rng) < 0.0) {
      angle = -angle;
    }

    vec3_t axis = random_unit_vector();
    vec3_t pivot = centroid(coords, indices);

    Eigen::AngleAxisd rot(angle, axis);
    for (auto i : indices) {
      vec3_t r = coords.row(i).transpose() - pivot;
      vec3_t r2 = rot * r;
      coords.row(i) = (r2 + pivot).transpose();
    }
  }

private:
  vec3_t random_unit_vector() {
    std::normal_distribution<double> nd(0.0, 1.0);
    vec3_t v = vec3_t::NullaryExpr([&] { return nd(rng); });
    const double n = v.norm();
    return (n < 1e-12) ? vec3_t{vec3_t::UnitX()} : v / n;
  }
};

// Rotates about a fixed axis (e.g. a bond vector).
struct RotationAboutAxisGenerator
    : MoveGeneratorBase<RotationAboutAxisGenerator> {
  vec3_t axis{0.0, 0.0, 1.0};
  double min_angle{0.0};
  double max_angle{0.1};
  mutable std::mt19937 rng;

  RotationAboutAxisGenerator() = default;
  RotationAboutAxisGenerator(vec3_t ax, double mn, double mx,
                             std::uint32_t seed = 42)
      : axis(ax.normalized()), min_angle(mn), max_angle(mx), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double angle =
        (min_angle < max_angle)
            ? std::uniform_real_distribution<double>(min_angle, max_angle)(rng)
            : min_angle;
    std::uniform_real_distribution<double> sign_dist(-1.0, 1.0);
    if (sign_dist(rng) < 0.0) {
      angle = -angle;
    }

    vec3_t pivot = centroid(coords, indices);
    Eigen::AngleAxisd rot(angle, axis);
    for (auto i : indices) {
      vec3_t r = coords.row(i).transpose() - pivot;
      coords.row(i) = (rot * r + pivot).transpose();
    }
  }
};

} // namespace fullrmc
