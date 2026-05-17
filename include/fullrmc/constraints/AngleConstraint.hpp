#pragma once
#include <fullrmc/constraints/Constraint.hpp>
#include <vector>
#include <cmath>

namespace fullrmc {

// Enforces bond-angle bounds for atom triplets (i–j–k), angle at j.
class AngleConstraint : public ConstraintBase<AngleConstraint> {
public:
    struct Triplet { index_t i, j, k; real_t lo, hi; };

    void add_angle(index_t i, index_t j, index_t k, real_t lo_rad, real_t hi_rad) {
        triplets_.push_back({i, j, k, lo_rad, hi_rad});
    }

    [[nodiscard]] std::string name() const override { return "AngleConstraint"; }

    [[nodiscard]] real_t compute_error(const coords_t& coords,
                                        std::span<const index_t> /*moved*/) const {
        real_t err = 0.0;
        for (auto& t : triplets_) {
            vec3_t v1 = (coords.row(t.i) - coords.row(t.j)).transpose();
            vec3_t v2 = (coords.row(t.k) - coords.row(t.j)).transpose();
            if (bc_) {
                v1 = bc_min_image(*bc_, v1);
                v2 = bc_min_image(*bc_, v2);
            }
            real_t cos_a = v1.dot(v2) / (v1.norm() * v2.norm() + 1e-30);
            cos_a = std::clamp(cos_a, -1.0, 1.0);
            real_t angle = std::acos(cos_a);
            if      (angle < t.lo) err += (t.lo - angle);
            else if (angle > t.hi) err += (angle - t.hi);
        }
        return err;
    }

private:
    std::vector<Triplet> triplets_;
};

} // namespace fullrmc
