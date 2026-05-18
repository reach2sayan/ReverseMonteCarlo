#pragma once
#include <Eigen/Geometry>
#include <RMC/core/RngGenerator.hpp>
#include <RMC/generators/MoveGenerator.hpp>

namespace RMC {

// Rotates all atoms in a group about a random axis through the group centroid.
// Angle is drawn uniformly from [min_angle, max_angle] (radians).
struct RotationGenerator : MoveGeneratorBase<RotationGenerator> {
  double min_angle{0.0};
  double max_angle{0.1}; // ~5.7 degrees
  mutable RngBuffer<> rng;

  RotationGenerator() = default;
  RotationGenerator(double mn, double mx, std::uint32_t seed = 42)
      : min_angle(mn), max_angle(mx), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double angle =
        (min_angle < max_angle) ? rng.uniform(min_angle, max_angle) : min_angle;
    angle = rng.uniform() < 0.5 ? angle : -angle;
    vec3_t axis = random_unit_vector();
    vec3_t pivot = centroid(coords, indices);
    const Eigen::Matrix3d R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    for (auto i : indices) {
      vec3_t r = coords.row(static_cast<Eigen::Index>(i)).transpose() - pivot;
      coords.row(static_cast<Eigen::Index>(i)) = (R * r + pivot).transpose();
    }
  }

private:
  FORCE_INLINE vec3_t random_unit_vector() {
    vec3_t v(rng.normal(), rng.normal(), rng.normal());
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
  mutable RngBuffer<> rng;

  RotationAboutAxisGenerator() = default;
  RotationAboutAxisGenerator(vec3_t ax, double mn, double mx,
                             std::uint32_t seed = 42)
      : axis(ax.normalized()), min_angle(mn), max_angle(mx), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double angle =
        (min_angle < max_angle) ? rng.uniform(min_angle, max_angle) : min_angle;
    angle *= rng.uniform() < 0.5 ? 1.0 : -1.0;

    vec3_t pivot = centroid(coords, indices);
    const Eigen::Matrix3d R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    for (auto i : indices) {
      vec3_t r = coords.row(static_cast<Eigen::Index>(i)).transpose() - pivot;
      coords.row(static_cast<Eigen::Index>(i)) = (R * r + pivot).transpose();
    }
  }
};

} // namespace RMC
