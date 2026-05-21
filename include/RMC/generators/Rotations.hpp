#pragma once
#include <Eigen/Geometry>
#include <RMC/core/RngGenerator.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <numbers>

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

  void generate(MoveGenerator::Token, coords_t &coords,
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

  void generate(MoveGenerator::Token, coords_t &coords,
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

// Rotates about one of the three Cartesian symmetry axes.
struct RotationAboutSymmetryAxisGenerator
    : MoveGeneratorBase<RotationAboutSymmetryAxisGenerator> {
  SymmetryAxis axis{SymmetryAxis::Z};
  double min_angle{0.0};
  double max_angle{0.1};
  mutable RngBuffer<> rng;

  RotationAboutSymmetryAxisGenerator() = default;
  RotationAboutSymmetryAxisGenerator(SymmetryAxis ax, double mn, double mx,
                                     std::uint32_t seed = 42)
      : axis(ax), min_angle(mn), max_angle(mx), rng(seed) {}

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double angle =
        (min_angle < max_angle) ? rng.uniform(min_angle, max_angle) : min_angle;
    angle *= rng.uniform() < 0.5 ? 1.0 : -1.0;
    vec3_t ax_vec = (axis == SymmetryAxis::X)   ? vec3_t::UnitX()
                    : (axis == SymmetryAxis::Y) ? vec3_t::UnitY()
                                                : vec3_t::UnitZ();
    vec3_t pivot = centroid(coords, indices);
    const Eigen::Matrix3d R =
        Eigen::AngleAxisd(angle, ax_vec).toRotationMatrix();
    for (auto i : indices) {
      vec3_t r = coords.row(static_cast<Eigen::Index>(i)).transpose() - pivot;
      coords.row(static_cast<Eigen::Index>(i)) = (R * r + pivot).transpose();
    }
  }
};

// Aligns the group's principal axis (indices.front() → indices.back()) toward
// `target_axis`, with a random angular perturbation up to `max_offset` radians.
struct OrientationGenerator : MoveGeneratorBase<OrientationGenerator> {
  vec3_t target_axis{0.0, 0.0, 1.0};
  double max_offset{0.1};
  mutable RngBuffer<> rng;

  OrientationGenerator() = default;
  OrientationGenerator(vec3_t target, double offset, std::uint32_t seed = 42)
      : target_axis(target.normalized()), max_offset(offset), rng(seed) {}

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    if (indices.size() < 2) {
      return;
    }
    const auto ia = static_cast<Eigen::Index>(indices.front());
    const auto ib = static_cast<Eigen::Index>(indices.back());
    vec3_t current = (coords.row(ib) - coords.row(ia)).transpose();
    if (current.norm() < 1e-12) {
      return;
    }
    current.normalize();

    // Add small random perturbation around the target axis.
    vec3_t perp = target_axis.unitOrthogonal();
    double offset = rng.uniform(0.0, max_offset);
    double phi = rng.uniform(0.0, 2.0 * std::numbers::pi);
    Eigen::AngleAxisd noise(offset, perp);
    Eigen::AngleAxisd spin(phi, target_axis);
    vec3_t desired = (spin * noise).toRotationMatrix() * target_axis;
    desired.normalize();

    // Rotation from current group axis to desired direction.
    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(current, desired);
    const Eigen::Matrix3d R = q.toRotationMatrix();
    vec3_t pivot = centroid(coords, indices);
    for (auto i : indices) {
      vec3_t r = coords.row(static_cast<Eigen::Index>(i)).transpose() - pivot;
      coords.row(static_cast<Eigen::Index>(i)) = (R * r + pivot).transpose();
    }
  }
};

} // namespace RMC
