#pragma once
#include <RMC/generators/MoveGenerator.hpp>
#include <optional>

namespace RMC {

namespace detail {
// Displacement of length `amp` from `from` towards `to`; nullopt when there.
[[nodiscard]] inline std::optional<vec3_t>
step_towards(const vec3_t &from, const vec3_t &to, double amp) {
  const vec3_t d = to - from;
  const double n = d.norm();
  return n < 1e-12 ? std::nullopt : std::optional<vec3_t>{d * (amp / n)};
}
} // namespace detail

// Translates the group by a [min, max] amplitude in a random direction.
struct TranslationGenerator : MoveGeneratorBase<TranslationGenerator> {
  Amplitude amp;
  Rng rng;

  TranslationGenerator() = default;
  TranslationGenerator(double mn, double mx, std::uint32_t seed = 42)
      : amp{mn, mx}, rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    const double a = amp.draw(rng);
    translate(coords, indices, a * random_unit_vector(rng));
  }
};

// Translates along a fixed axis by a signed amplitude.
struct TranslationAlongAxisGenerator
    : MoveGeneratorBase<TranslationAlongAxisGenerator> {
  vec3_t axis{vec3_t::UnitX()};
  Amplitude amp;
  Rng rng;

  TranslationAlongAxisGenerator() = default;
  TranslationAlongAxisGenerator(const vec3_t &ax, double mn, double mx,
                                std::uint32_t seed = 42)
      : axis(ax.normalized()), amp{mn, mx}, rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    translate(coords, indices, amp.signed_draw(rng) * axis);
  }
};

// Along one of the three Cartesian axes.
struct TranslationAlongSymmetryAxisGenerator : TranslationAlongAxisGenerator {
  TranslationAlongSymmetryAxisGenerator(SymmetryAxis ax, double mn, double mx,
                                        std::uint32_t seed = 42)
      : TranslationAlongAxisGenerator(unit(ax), mn, mx, seed) {}
};

// Translates towards a fixed centre point.
struct TranslationTowardsCentreGenerator
    : MoveGeneratorBase<TranslationTowardsCentreGenerator> {
  vec3_t centre{vec3_t::Zero()};
  Amplitude amp;
  Rng rng;

  TranslationTowardsCentreGenerator() = default;
  TranslationTowardsCentreGenerator(const vec3_t &c, double mn, double mx,
                                    std::uint32_t seed = 42)
      : centre(c), amp{mn, mx}, rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    const double a = amp.draw(rng);
    if (const auto d = detail::step_towards(centroid(coords, indices), centre, a)) {
      translate(coords, indices, *d);
    }
  }
};

// Translates toward the nearest point on the line through `point` along
// `direction`.
struct TranslationTowardsAxisGenerator
    : MoveGeneratorBase<TranslationTowardsAxisGenerator> {
  vec3_t point{vec3_t::Zero()};
  vec3_t direction{vec3_t::UnitZ()};
  Amplitude amp;
  Rng rng;

  TranslationTowardsAxisGenerator() = default;
  TranslationTowardsAxisGenerator(const vec3_t &pt, const vec3_t &dir,
                                  double mn, double mx, std::uint32_t seed = 42)
      : point(pt), direction(dir.normalized()), amp{mn, mx}, rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    const double a = amp.draw(rng);
    const vec3_t gc = centroid(coords, indices);
    const vec3_t nearest = point + direction.dot(gc - point) * direction;
    if (const auto d = detail::step_towards(gc, nearest, a)) {
      translate(coords, indices, *d);
    }
  }
};

// Towards a Cartesian axis through the origin.
struct TranslationTowardsSymmetryAxisGenerator
    : TranslationTowardsAxisGenerator {
  TranslationTowardsSymmetryAxisGenerator(SymmetryAxis ax, double mn, double mx,
                                          std::uint32_t seed = 42)
      : TranslationTowardsAxisGenerator(vec3_t::Zero(), unit(ax), mn, mx, seed) {}
};

} // namespace RMC
