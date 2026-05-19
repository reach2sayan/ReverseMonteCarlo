#pragma once
#include <Eigen/Geometry>
#include <RMC/core/RngGenerator.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/assert.hpp>

namespace RMC {

// Pushes/pulls two atoms symmetrically along their bond direction by ±amp/2 each,
// preserving the bond midpoint. Atoms `i` and `j` must be present in `indices`.
struct DistanceAgitationGenerator
    : MoveGeneratorBase<DistanceAgitationGenerator> {
  std::size_t i{0};
  std::size_t j{1};
  double min_amp{0.0};
  double max_amp{0.05};
  mutable RngBuffer<> rng;

  DistanceAgitationGenerator() = default;
  DistanceAgitationGenerator(std::size_t ai, std::size_t aj, double mn,
                              double mx, std::uint32_t seed = 42)
      : i(ai), j(aj), min_amp(mn), max_amp(mx), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> /*indices*/) {
    double amp = (min_amp < max_amp) ? rng.uniform(min_amp, max_amp) : min_amp;
    if (rng.uniform() < 0.5)
      amp = -amp;
    const auto ei = static_cast<Eigen::Index>(i);
    const auto ej = static_cast<Eigen::Index>(j);
    vec3_t bond = (coords.row(ej) - coords.row(ei)).transpose();
    double len = bond.norm();
    if (len < 1e-12)
      return;
    vec3_t bond_dir = bond / len;
    coords.row(ei) -= (0.5 * amp * bond_dir).transpose();
    coords.row(ej) += (0.5 * amp * bond_dir).transpose();
  }
};

// Adjusts the angle at vertex `j` for triplet (i,j,k) while keeping bond
// lengths |i-j| and |k-j| constant. Rotates i by -amp/2 and k by +amp/2
// about the axis perpendicular to the i-j-k plane through j.
struct AngleAgitationGenerator : MoveGeneratorBase<AngleAgitationGenerator> {
  std::size_t i{0};
  std::size_t j{1};
  std::size_t k{2};
  double min_amp{0.0};
  double max_amp{0.05};
  mutable RngBuffer<> rng;

  AngleAgitationGenerator() = default;
  AngleAgitationGenerator(std::size_t ai, std::size_t aj, std::size_t ak,
                           double mn, double mx, std::uint32_t seed = 42)
      : i(ai), j(aj), k(ak), min_amp(mn), max_amp(mx), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> /*indices*/) {
    double amp = (min_amp < max_amp) ? rng.uniform(min_amp, max_amp) : min_amp;
    if (rng.uniform() < 0.5)
      amp = -amp;
    const auto ei = static_cast<Eigen::Index>(i);
    const auto ej = static_cast<Eigen::Index>(j);
    const auto ek = static_cast<Eigen::Index>(k);
    vec3_t pj = coords.row(ej).transpose();
    vec3_t ji = (coords.row(ei) - coords.row(ej)).transpose();
    vec3_t jk = (coords.row(ek) - coords.row(ej)).transpose();
    // Rotation axis perpendicular to the i-j-k plane.
    vec3_t rot_axis = ji.cross(jk);
    if (rot_axis.norm() < 1e-12)
      return;
    rot_axis.normalize();
    // Rotate i by -amp/2 about rot_axis through j.
    Eigen::Matrix3d Ri =
        Eigen::AngleAxisd(-0.5 * amp, rot_axis).toRotationMatrix();
    coords.row(ei) = (Ri * ji + pj).transpose();
    // Rotate k by +amp/2 about rot_axis through j.
    Eigen::Matrix3d Rk =
        Eigen::AngleAxisd(+0.5 * amp, rot_axis).toRotationMatrix();
    coords.row(ek) = (Rk * jk + pj).transpose();
  }
};

} // namespace RMC
