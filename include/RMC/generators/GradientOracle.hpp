#pragma once
#include <Eigen/Geometry>
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/core/Types.hpp>
#include <ranges>
#include <span>

namespace RMC {

// Central-difference gradient oracle shared by all Langevin/Leapfrog
// generators. Calls constraints.compute_after_move() twice per parameter
// dimension — the Engine overwrites err_after_ again in its own score_after
// stage, so these calls are safe.
struct GradientOracle {
  static constexpr double default_fd_step = 1e-4;
  static constexpr double rotation_fd_step = 1e-5;

  // Returns ∂χ²/∂coords for Cartesian coordinates of the group atoms.
  // Result: 3k-vector [∂χ²/∂x₁, ∂χ²/∂y₁, ∂χ²/∂z₁, ..., ∂χ²/∂z_k]
  static vec_t translation_gradient(coords_t &coords,
                                    std::span<const std::size_t> indices,
                                    ConstraintCollection &constraints,
                                    double fd_step = default_fd_step) {
    const std::size_t k = indices.size();
    vec_t grad(static_cast<Eigen::Index>(3 * k));
    for (const auto [ai, atom] : indices | std::views::enumerate) {
      for (Eigen::Index ax : {0, 1, 2}) {
        auto x = coords(atom, ax);

        coords(atom, ax) = x + fd_step;
        constraints.compute_after_move(coords, indices);
        const double err_plus = constraints.total_error();

        coords(atom, ax) = x - fd_step;
        constraints.compute_after_move(coords, indices);
        const double err_minus = constraints.total_error();

        coords(atom, ax) = x;
        grad(3 * ai + ax) = (err_plus - err_minus) / (2.0 * fd_step);
      }
    }
    return grad;
  }

  // Returns ∂χ²/∂θ for rotation by angle θ about `axis` through `pivot`.
  // Saves and restores atom positions — leaves coords unchanged on exit.
  static double rotation_gradient(coords_t &coords,
                                  std::span<const std::size_t> indices,
                                  const vec3_t &axis, const vec3_t &pivot,
                                  ConstraintCollection &constraints,
                                  double fd_step = rotation_fd_step) {
    const std::size_t k = indices.size();
    Eigen::Map<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> idx(
        indices.data(), static_cast<Eigen::Index>(k));
    Eigen::MatrixXd saved = coords(idx, Eigen::all);

    auto apply_rot = [&](double theta) {
      Eigen::AngleAxisd rot(theta, axis);
      for (auto ai = 0u; ai < k; ++ai) {
        const auto atom = static_cast<Eigen::Index>(indices[ai]);
        coords.row(atom) =
            (rot * (saved.row(static_cast<Eigen::Index>(ai)).transpose() -
                    pivot) +
             pivot)
                .transpose();
      }
    };

    apply_rot(+fd_step);
    constraints.compute_after_move(coords, indices);
    const double err_plus = constraints.total_error();

    apply_rot(-fd_step);
    constraints.compute_after_move(coords, indices);
    const double err_minus = constraints.total_error();

    // Restore original positions
    coords(idx, Eigen::all) = saved;
    return (err_plus - err_minus) / (2.0 * fd_step);
  }
};

} // namespace RMC
