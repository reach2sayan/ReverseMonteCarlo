#pragma once
#include <Eigen/Geometry>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/assert.hpp>
#include <vector>

namespace RMC {

// Applies a predefined sequence of signed displacements along a fixed axis,
// cycling through `path` on each successive generate() call.
struct TranslationAlongAxisPath : MoveGeneratorBase<TranslationAlongAxisPath> {
  vec3_t axis{0.0, 0.0, 1.0};
  std::vector<double> path;

  TranslationAlongAxisPath() = default;
  TranslationAlongAxisPath(vec3_t ax, std::vector<double> displacements)
      : axis(ax.normalized()), path(std::move(displacements)) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    if (path.empty())
      return;
    double magnitude = path[step_++ % path.size()];
    vec3_t delta = axis * magnitude;
    coords(indices, Eigen::all).rowwise() += delta.transpose();
  }

private:
  mutable std::size_t step_{0};
};

// Applies a predefined sequence of rotation angles about a fixed axis,
// cycling through `path` on each successive generate() call.
// The pivot is always the group centroid.
struct RotationAboutAxisPath : MoveGeneratorBase<RotationAboutAxisPath> {
  vec3_t axis{0.0, 0.0, 1.0};
  std::vector<double> path;

  RotationAboutAxisPath() = default;
  RotationAboutAxisPath(vec3_t ax, std::vector<double> angles)
      : axis(ax.normalized()), path(std::move(angles)) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    if (path.empty())
      return;
    double angle = path[step_++ % path.size()];
    vec3_t pivot = centroid(coords, indices);
    const Eigen::Matrix3d R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    for (auto i : indices) {
      vec3_t r = coords.row(static_cast<Eigen::Index>(i)).transpose() - pivot;
      coords.row(static_cast<Eigen::Index>(i)) = (R * r + pivot).transpose();
    }
  }

private:
  mutable std::size_t step_{0};
};

} // namespace RMC
