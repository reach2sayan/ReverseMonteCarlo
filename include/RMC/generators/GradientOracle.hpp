#pragma once
#include <Eigen/Geometry>
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/core/Types.hpp>
#include <span>

namespace RMC {

// Central-difference gradient oracle shared by all Langevin/Leapfrog generators.
// Calls constraints.compute_after_move() twice per parameter dimension — the Engine
// overwrites err_after_ again in its own score_after stage, so these calls are safe.
struct GradientOracle {
    static constexpr double default_fd_step = 1e-4;
    static constexpr double rotation_fd_step = 1e-5;

    // Returns ∂χ²/∂coords for Cartesian coordinates of the group atoms.
    // Result: 3k-vector [∂χ²/∂x₁, ∂χ²/∂y₁, ∂χ²/∂z₁, ..., ∂χ²/∂z_k]
    static vec_t translation_gradient(
        coords_t &coords,
        std::span<const std::size_t> indices,
        ConstraintCollection &constraints,
        double fd_step = default_fd_step)
    {
        const std::size_t k = indices.size();
        vec_t grad(static_cast<Eigen::Index>(3 * k));

        for (std::size_t ai = 0; ai < k; ++ai) {
            const Eigen::Index atom = static_cast<Eigen::Index>(indices[ai]);
            for (int ax = 0; ax < 3; ++ax) {
                const double orig = coords(atom, ax);

                coords(atom, ax) = orig + fd_step;
                constraints.compute_after_move(coords, indices);
                const double err_plus = constraints.total_error();

                coords(atom, ax) = orig - fd_step;
                constraints.compute_after_move(coords, indices);
                const double err_minus = constraints.total_error();

                coords(atom, ax) = orig;
                grad[static_cast<Eigen::Index>(3 * ai + static_cast<std::size_t>(ax))] =
                    (err_plus - err_minus) / (2.0 * fd_step);
            }
        }
        return grad;
    }

    // Returns ∂χ²/∂θ for rotation by angle θ about `axis` through `pivot`.
    // Saves and restores atom positions — leaves coords unchanged on exit.
    static double rotation_gradient(
        coords_t &coords,
        std::span<const std::size_t> indices,
        const vec3_t &axis,
        const vec3_t &pivot,
        ConstraintCollection &constraints,
        double fd_step = rotation_fd_step)
    {
        const std::size_t k = indices.size();
        Eigen::MatrixXd saved(static_cast<Eigen::Index>(k), 3);
        for (std::size_t ai = 0; ai < k; ++ai)
            saved.row(static_cast<Eigen::Index>(ai)) =
                coords.row(static_cast<Eigen::Index>(indices[ai]));

        auto apply_rot = [&](double theta) {
            Eigen::AngleAxisd rot(theta, axis);
            for (std::size_t ai = 0; ai < k; ++ai) {
                vec3_t r = saved.row(static_cast<Eigen::Index>(ai)).transpose() - pivot;
                coords.row(static_cast<Eigen::Index>(indices[ai])) =
                    (rot * r + pivot).transpose();
            }
        };

        apply_rot(+fd_step);
        constraints.compute_after_move(coords, indices);
        const double err_plus = constraints.total_error();

        apply_rot(-fd_step);
        constraints.compute_after_move(coords, indices);
        const double err_minus = constraints.total_error();

        // Restore original positions
        for (std::size_t ai = 0; ai < k; ++ai)
            coords.row(static_cast<Eigen::Index>(indices[ai])) =
                saved.row(static_cast<Eigen::Index>(ai));

        return (err_plus - err_minus) / (2.0 * fd_step);
    }
};

} // namespace RMC
