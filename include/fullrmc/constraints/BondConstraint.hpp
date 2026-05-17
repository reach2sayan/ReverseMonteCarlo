#pragma once
#include <fullrmc/constraints/Constraint.hpp>
#include <map>
#include <utility>
#include <algorithm>
#include <cmath>

namespace fullrmc {

// Enforces bond-length bounds for explicit atom-index pairs.
// std_err = Σ violations where violation > 0 means bond is outside [lo, hi].
class BondConstraint : public ConstraintBase<BondConstraint> {
public:
    struct Bound { real_t lo, hi; };
    // Pair key is always (min(i,j), max(i,j)).
    using PairKey = std::pair<index_t, index_t>;

    void add_bond(index_t i, index_t j, real_t lo, real_t hi) {
        bonds_[{std::min(i,j), std::max(i,j)}] = {lo, hi};
    }

    [[nodiscard]] std::string name() const override { return "BondConstraint"; }

    [[nodiscard]] real_t compute_error(const coords_t& coords,
                                        std::span<const index_t> /*moved*/) const {
        real_t err = 0.0;
        for (auto& [key, bnd] : bonds_) {
            real_t d = distance(coords, key.first, key.second);
            if      (d < bnd.lo) err += (bnd.lo - d);
            else if (d > bnd.hi) err += (d - bnd.hi);
        }
        return err;
    }

private:
    std::map<PairKey, Bound> bonds_;
};

} // namespace fullrmc
