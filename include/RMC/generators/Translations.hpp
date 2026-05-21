#pragma once
#include <RMC/core/RngGenerator.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <cmath>

namespace RMC {

// Translates all atoms in a group by the same random displacement vector.
// Amplitude is drawn uniformly from [min_amp, max_amp] (Angstrom).
// Direction is uniform on the unit sphere via the Marsaglia method.
struct TranslationGenerator : MoveGeneratorBase<TranslationGenerator> {
  double min_amp{0.0};
  double max_amp{0.2};
  mutable RngBuffer<> rng;

  TranslationGenerator() = default;
  TranslationGenerator(double mn, double mx, std::uint32_t seed = 42)
      : min_amp(mn), max_amp(mx), rng(seed) {}

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double amp = (min_amp < max_amp) ? rng.uniform(min_amp, max_amp) : min_amp;
    vec3_t delta = random_unit_vector() * amp;
    coords(indices, Eigen::all).rowwise() += delta.transpose();
  }

private:
  FORCE_INLINE vec3_t random_unit_vector() {
    // Marsaglia (1972) uniform sphere sampling
    vec3_t v(rng.normal(), rng.normal(), rng.normal());
    double n = v.norm();
    if (n < 1e-12) {
      return vec3_t::UnitX();
    }
    return v / n;
  }
};

// Translates along a fixed axis only.
struct TranslationAlongAxisGenerator
    : MoveGeneratorBase<TranslationAlongAxisGenerator> {
  vec3_t axis{1.0, 0.0, 0.0};
  double min_amp{0.0};
  double max_amp{0.2};
  mutable RngBuffer<> rng;

  TranslationAlongAxisGenerator() = default;
  TranslationAlongAxisGenerator(vec3_t ax, double mn, double mx,
                                std::uint32_t seed = 42)
      : axis(ax.normalized()), min_amp(mn), max_amp(mx), rng(seed) {}

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double magnitude =
        (min_amp < max_amp) ? rng.uniform(min_amp, max_amp) : min_amp;
    if (rng.uniform() < 0.5)
      magnitude = -magnitude;
    vec3_t delta = axis * magnitude;
    coords(indices, Eigen::all).rowwise() += delta.transpose();
  }
};

// Translates towards a fixed centre point.
struct TranslationTowardsCentreGenerator
    : MoveGeneratorBase<TranslationTowardsCentreGenerator> {
  vec3_t centre{0.0, 0.0, 0.0};
  double min_amp{0.0};
  double max_amp{0.2};
  mutable RngBuffer<> rng;

  TranslationTowardsCentreGenerator() = default;
  TranslationTowardsCentreGenerator(vec3_t c, double mn, double mx,
                                    std::uint32_t seed = 42)
      : centre(c), min_amp(mn), max_amp(mx), rng(seed) {}

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double amp = (min_amp < max_amp) ? rng.uniform(min_amp, max_amp) : min_amp;
    vec3_t gc = centroid(coords, indices);
    vec3_t dir = (centre - gc);
    double dist = dir.norm();
    if (dist < 1e-12) {
      return;
    }
    vec3_t delta = (dir / dist) * amp;
    coords(indices, Eigen::all).rowwise() += delta.transpose();
  }
};

// Translates toward the nearest point on an infinite axis (line through `point`
// along `direction`).
struct TranslationTowardsAxisGenerator
    : MoveGeneratorBase<TranslationTowardsAxisGenerator> {
  vec3_t point{0.0, 0.0, 0.0};
  vec3_t direction{0.0, 0.0, 1.0};
  double min_amp{0.0};
  double max_amp{0.2};
  mutable RngBuffer<> rng;

  TranslationTowardsAxisGenerator() = default;
  TranslationTowardsAxisGenerator(vec3_t pt, vec3_t dir, double mn, double mx,
                                  std::uint32_t seed = 42)
      : point(std::move(pt)), direction(dir.normalized()), min_amp(mn),
        max_amp(mx), rng(seed) {}

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double amp = (min_amp < max_amp) ? rng.uniform(min_amp, max_amp) : min_amp;
    vec3_t gc = centroid(coords, indices);
    vec3_t diff = gc - point;
    vec3_t nearest = point + direction.dot(diff) * direction;
    vec3_t to_axis = nearest - gc;
    double dist = to_axis.norm();
    if (dist < 1e-12)
      return;
    vec3_t delta = (to_axis / dist) * amp;
    coords(indices, Eigen::all).rowwise() += delta.transpose();
  }
};

// Translates along one of the three Cartesian symmetry axes.
struct TranslationAlongSymmetryAxisGenerator
    : MoveGeneratorBase<TranslationAlongSymmetryAxisGenerator> {
  SymmetryAxis axis{SymmetryAxis::Z};
  double min_amp{0.0};
  double max_amp{0.2};
  mutable RngBuffer<> rng;

  TranslationAlongSymmetryAxisGenerator() = default;
  TranslationAlongSymmetryAxisGenerator(SymmetryAxis ax, double mn, double mx,
                                        std::uint32_t seed = 42)
      : axis(ax), min_amp(mn), max_amp(mx), rng(seed) {}

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double magnitude =
        (min_amp < max_amp) ? rng.uniform(min_amp, max_amp) : min_amp;
    if (rng.uniform() < 0.5)
      magnitude = -magnitude;
    vec3_t delta = vec3_t::Zero();
    if (axis == SymmetryAxis::X)
      delta.x() = magnitude;
    else if (axis == SymmetryAxis::Y)
      delta.y() = magnitude;
    else
      delta.z() = magnitude;
    coords(indices, Eigen::all).rowwise() += delta.transpose();
  }
};

// Translates toward the nearest point on a Cartesian symmetry axis through the
// origin.
struct TranslationTowardsSymmetryAxisGenerator
    : MoveGeneratorBase<TranslationTowardsSymmetryAxisGenerator> {
  SymmetryAxis axis{SymmetryAxis::Z};
  double min_amp{0.0};
  double max_amp{0.2};
  mutable RngBuffer<> rng;

  TranslationTowardsSymmetryAxisGenerator() = default;
  TranslationTowardsSymmetryAxisGenerator(SymmetryAxis ax, double mn, double mx,
                                          std::uint32_t seed = 42)
      : axis(ax), min_amp(mn), max_amp(mx), rng(seed) {}

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double amp = (min_amp < max_amp) ? rng.uniform(min_amp, max_amp) : min_amp;
    vec3_t gc = centroid(coords, indices);
    // Nearest point on the symmetry axis: zero out the two perpendicular
    // components.
    vec3_t nearest = gc;
    if (axis == SymmetryAxis::X) {
      nearest.y() = 0.0;
      nearest.z() = 0.0;
    } else if (axis == SymmetryAxis::Y) {
      nearest.x() = 0.0;
      nearest.z() = 0.0;
    } else {
      nearest.x() = 0.0;
      nearest.y() = 0.0;
    }
    vec3_t to_axis = nearest - gc;
    double dist = to_axis.norm();
    if (dist < 1e-12)
      return;
    vec3_t delta = (to_axis / dist) * amp;
    coords(indices, Eigen::all).rowwise() += delta.transpose();
  }
};

} // namespace RMC
