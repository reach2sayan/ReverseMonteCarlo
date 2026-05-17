#pragma once
#include <fullrmc/constraints/Constraint.hpp>
#include <vector>
#include <cmath>

namespace fullrmc {

// Improper dihedral: enforces planarity / chirality for quadruplets (i,j,k,l)
// where the angle is between plane(i,j,k) and plane(j,k,l).
class ImproperAngleConstraint : public ConstraintBase<ImproperAngleConstraint> {
public:
    struct Quad { index_t i, j, k, l; real_t lo, hi; };

    void add_improper(index_t i, index_t j, index_t k, index_t l,
                      real_t lo_rad, real_t hi_rad) {
        quads_.push_back({i, j, k, l, lo_rad, hi_rad});
    }

    [[nodiscard]] std::string name() const override { return "ImproperAngleConstraint"; }

    [[nodiscard]] real_t compute_error(const coords_t& coords,
                                        std::span<const index_t> /*moved*/) const {
        real_t err = 0.0;
        for (auto& q : quads_) {
            real_t phi = improper(coords, q.i, q.j, q.k, q.l);
            if      (phi < q.lo) err += (q.lo - phi);
            else if (phi > q.hi) err += (phi - q.hi);
        }
        return err;
    }

private:
    std::vector<Quad> quads_;

    // Same formula as dihedral – improper is just a dihedral with a
    // different atom ordering chosen to measure out-of-plane distortion.
    [[nodiscard]] real_t improper(const coords_t& c,
                                   index_t i, index_t j,
                                   index_t k, index_t l) const noexcept {
        vec3_t b1 = c.row(j).transpose() - c.row(i).transpose();
        vec3_t b2 = c.row(k).transpose() - c.row(j).transpose();
        vec3_t b3 = c.row(l).transpose() - c.row(k).transpose();
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
