#pragma once
#include <RMC/generators/MoveGenerator.hpp>
#include <numbers>

namespace RMC {

// Rotates the group about a random axis through its centroid by a signed
// angle drawn from [min, max] (radians).
struct RotationGenerator : MoveGeneratorBase<RotationGenerator> {
  Amplitude angle{0.0, 0.1};
  Rng rng;

  RotationGenerator() = default;
  RotationGenerator(double mn, double mx, std::uint32_t seed = 42)
      : angle{mn, mx}, rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    const double a = angle.signed_draw(rng);
    rotate_group(coords, indices, a, random_unit_vector(rng));
  }
};

// Rotates about a fixed axis (e.g. a bond vector) through the centroid.
struct RotationAboutAxisGenerator
    : MoveGeneratorBase<RotationAboutAxisGenerator> {
  vec3_t axis{vec3_t::UnitZ()};
  Amplitude angle{0.0, 0.1};
  Rng rng;

  RotationAboutAxisGenerator() = default;
  RotationAboutAxisGenerator(const vec3_t &ax, double mn, double mx,
                             std::uint32_t seed = 42)
      : axis(ax.normalized()), angle{mn, mx}, rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    rotate_group(coords, indices, angle.signed_draw(rng), axis);
  }
};

// About one of the three Cartesian axes.
struct RotationAboutSymmetryAxisGenerator : RotationAboutAxisGenerator {
  RotationAboutSymmetryAxisGenerator(SymmetryAxis ax, double mn, double mx,
                                     std::uint32_t seed = 42)
      : RotationAboutAxisGenerator(unit(ax), mn, mx, seed) {}
};

// Aligns the group's principal axis (indices.front() → indices.back()) toward
// `target_axis`, tilted by a random angle up to `max_offset` radians.
struct OrientationGenerator : MoveGeneratorBase<OrientationGenerator> {
  vec3_t target_axis{vec3_t::UnitZ()};
  double max_offset{0.1};
  Rng rng;

  OrientationGenerator() = default;
  OrientationGenerator(const vec3_t &target, double offset,
                       std::uint32_t seed = 42)
      : target_axis(target.normalized()), max_offset(offset), rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    if (indices.size() < 2) {
      return;
    }
    const vec3_t current =
        (coords.row(indices.back()) - coords.row(indices.front())).transpose();
    if (current.norm() < 1e-12) {
      return;
    }
    // Tilt the target about a perpendicular, then spin it about itself.
    const double offset = rng.uniform(0.0, max_offset);
    const double phi = rng.uniform(0.0, 2.0 * std::numbers::pi);
    const vec3_t desired =
        (Eigen::AngleAxisd(phi, target_axis) *
         Eigen::AngleAxisd(offset, target_axis.unitOrthogonal())) *
        target_axis;
    rotate_about(coords, indices,
                 Eigen::Quaterniond::FromTwoVectors(current, desired)
                     .toRotationMatrix(),
                 centroid(coords, indices));
  }
};

} // namespace RMC
