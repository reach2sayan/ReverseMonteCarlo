#pragma once
#include <RMC/generators/MoveGenerator.hpp>
#include <cmath>
#include <numbers>
#include <random>

namespace RMC {

// Translates all atoms in a group by the same random displacement vector.
// Amplitude is drawn uniformly from [min_amp, max_amp] (Angstrom).
// Direction is uniform on the unit sphere via the Marsaglia method.
struct TranslationGenerator : MoveGeneratorBase<TranslationGenerator> {
  double min_amp{0.0};
  double max_amp{0.2};
  mutable std::mt19937 rng;

  TranslationGenerator() = default;
  TranslationGenerator(double mn, double mx, std::uint32_t seed = 42)
      : min_amp(mn), max_amp(mx), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double amp =
        (min_amp < max_amp)
            ? std::uniform_real_distribution<double>(min_amp, max_amp)(rng)
            : min_amp;
    vec3_t delta = random_unit_vector() * amp;
    coords(indices, Eigen::all).rowwise() += delta.transpose();
  }

private:
  vec3_t random_unit_vector() {
    // Marsaglia (1972) uniform sphere sampling
    std::normal_distribution<double> nd(0.0, 1.0);
    vec3_t v(nd(rng), nd(rng), nd(rng));
    double n = v.norm();
    if (n < 1e-12)
      return vec3_t::UnitX();
    return v / n;
  }
};

// Translates along a fixed axis only.
struct TranslationAlongAxisGenerator
    : MoveGeneratorBase<TranslationAlongAxisGenerator> {
  vec3_t axis{1.0, 0.0, 0.0};
  double min_amp{0.0};
  double max_amp{0.2};
  mutable std::mt19937 rng;

  TranslationAlongAxisGenerator() = default;
  TranslationAlongAxisGenerator(vec3_t ax, double mn, double mx,
                                std::uint32_t seed = 42)
      : axis(ax.normalized()), min_amp(mn), max_amp(mx), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double magnitude =
        (min_amp < max_amp)
            ? std::uniform_real_distribution<double>(min_amp, max_amp)(rng)
            : min_amp;
    std::uniform_real_distribution<double> sign_dist(-1.0, 1.0);
    if (sign_dist(rng) < 0.0) {
      magnitude = -magnitude;
    }
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
  mutable std::mt19937 rng;

  TranslationTowardsCentreGenerator() = default;
  TranslationTowardsCentreGenerator(vec3_t c, double mn, double mx,
                                    std::uint32_t seed = 42)
      : centre(c), min_amp(mn), max_amp(mx), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    double amp =
        (min_amp < max_amp)
            ? std::uniform_real_distribution<double>(min_amp, max_amp)(rng)
            : min_amp;
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

} // namespace RMC
