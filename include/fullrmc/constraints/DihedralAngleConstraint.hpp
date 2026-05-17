#pragma once
#include <fullrmc/constraints/Constraint.hpp>
#include <vector>
#include <cmath>

namespace fullrmc {

// Enforces dihedral-angle bounds for atom quadruplets (i–j–k–l).
class DihedralAngleConstraint : public ConstraintBase<DihedralAngleConstraint> {
public:
    struct Quad { index_t i, j, k, l; real_t lo, hi; };

    void add_dihedral(index_t i, index_t j, index_t k, index_t l,
                      real_t lo_rad, real_t hi_rad) {
        quads_.push_back({i, j, k, l, lo_rad, hi_rad});
    }

    [[nodiscard]] std::string name() const override { return "DihedralAngleConstraint"; }

    [[nodiscard]] real_t compute_error(const coords_t& coords,
                                        std::span<const index_t> /*moved*/) const {
        real_t err = 0.0;
        for (auto& q : quads_) {
            real_t phi = dihedral(coords, q.i, q.j, q.k, q.l);
            if      (phi < q.lo) err += (q.lo - phi);
            else if (phi > q.hi) err += (phi - q.hi);
        }
        return err;
    }

private:
    std::vector<Quad> quads_;

    [[nodiscard]] real_t dihedral(const coords_t& c,
                                   index_t i, index_t j,
                                   index_t k, index_t l) const noexcept {
        auto get = [&](index_t a) -> vec3_t {
            return c.row(a).transpose();
        };
        vec3_t b1 = get(j) - get(i);
        vec3_t b2 = get(k) - get(j);
        vec3_t b3 = get(l) - get(k);
        if (bc_) {
            b1 = bc_min_image(*bc_, b1);
            b2 = bc_min_image(*bc_, b2);
            b3 = bc_min_image(*bc_, b3);
        }
        vec3_t n1 = b1.cross(b2);
        vec3_t n2 = b2.cross(b3);
        real_t x  = n1.dot(n2);
        real_t y  = (n1.cross(n2)).dot(b2.normalized());
        return std::atan2(y, x);
    }
};

} // namespace fullrmc
