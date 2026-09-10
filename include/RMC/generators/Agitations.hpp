#pragma once
#include <RMC/generators/MoveGenerator.hpp>

namespace RMC {

// Pushes/pulls atoms `i` and `j` symmetrically along their bond by ±amp/2
// each, preserving the bond midpoint.
struct DistanceAgitationGenerator
    : MoveGeneratorBase<DistanceAgitationGenerator> {
  std::size_t i{0};
  std::size_t j{1};
  Amplitude amp{0.0, 0.05};
  Rng rng;

  DistanceAgitationGenerator() = default;
  DistanceAgitationGenerator(std::size_t ai, std::size_t aj, double mn,
                             double mx, std::uint32_t seed = 42)
      : i(ai), j(aj), amp{mn, mx}, rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> /*indices*/) {
    const double a = amp.signed_draw(rng);
    const vec3_t bond = (coords.row(j) - coords.row(i)).transpose();
    const double len = bond.norm();
    if (len < 1e-12) {
      return;
    }
    const vec3_t half = (0.5 * a / len) * bond;
    coords.row(i) -= half.transpose();
    coords.row(j) += half.transpose();
  }
};

// Opens/closes the angle at vertex `j` of (i, j, k) keeping both bond lengths:
// i and k turn by ∓amp/2 about the normal of the i-j-k plane through j.
struct AngleAgitationGenerator : MoveGeneratorBase<AngleAgitationGenerator> {
  std::size_t i{0};
  std::size_t j{1};
  std::size_t k{2};
  Amplitude amp{0.0, 0.05};
  Rng rng;

  AngleAgitationGenerator() = default;
  AngleAgitationGenerator(std::size_t ai, std::size_t aj, std::size_t ak,
                          double mn, double mx, std::uint32_t seed = 42)
      : i(ai), j(aj), k(ak), amp{mn, mx}, rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> /*indices*/) {
    const double a = amp.signed_draw(rng);
    const vec3_t pj = coords.row(j).transpose();
    const vec3_t ji = coords.row(i).transpose() - pj;
    const vec3_t jk = coords.row(k).transpose() - pj;
    const vec3_t normal = ji.cross(jk);
    if (normal.norm() < 1e-12) {
      return;
    }
    const vec3_t axis = normal.normalized();
    coords.row(i) = (Eigen::AngleAxisd(-0.5 * a, axis) * ji + pj).transpose();
    coords.row(k) = (Eigen::AngleAxisd(+0.5 * a, axis) * jk + pj).transpose();
  }
};

} // namespace RMC
