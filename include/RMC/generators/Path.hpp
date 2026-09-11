#pragma once
#include <RMC/generators/MoveGenerator.hpp>
#include <utility>
#include <vector>

namespace RMC {

// Applies a predefined sequence of signed displacements along a fixed axis,
// cycling through `path` on each successive generate() call.
struct TranslationAlongAxisPath : MoveGeneratorBase<TranslationAlongAxisPath> {
  vec3_t axis{vec3_t::UnitZ()};
  std::vector<double> path;

  TranslationAlongAxisPath() = default;
  TranslationAlongAxisPath(const vec3_t &ax, std::vector<double> displacements)
      : axis(ax.normalized()), path(std::move(displacements)) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    if (!path.empty()) {
      translate(coords, indices, path[step_++ % path.size()] * axis);
    }
  }

private:
  std::size_t step_{0};
};

// Applies a predefined sequence of rotation angles about a fixed axis through
// the group centroid, cycling through `path` on each generate() call.
struct RotationAboutAxisPath : MoveGeneratorBase<RotationAboutAxisPath> {
  vec3_t axis{vec3_t::UnitZ()};
  std::vector<double> path;

  RotationAboutAxisPath() = default;
  RotationAboutAxisPath(const vec3_t &ax, std::vector<double> angles)
      : axis(ax.normalized()), path(std::move(angles)) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    if (!path.empty()) {
      rotate_group(coords, indices, path[step_++ % path.size()], axis);
    }
  }

private:
  std::size_t step_{0};
};

} // namespace RMC
