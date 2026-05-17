#pragma once
#include <algorithm>
#include <cmath>
#include <fullrmc/constraints/Constraint.hpp>
#include <map>
#include <utility>

namespace fullrmc {

// Enforces bond-length bounds for explicit atom-index pairs.
// std_err = Σ violations where violation > 0 means bond is outside [lo, hi].
class BondConstraint : public ConstraintBase<BondConstraint> {
public:
  struct Bound {
    double lo, hi;
  };
  // Pair key is always (min(i,j), max(i,j)).
  using PairKey = std::pair<std::size_t, std::size_t>;

  void add_bond(std::size_t i, std::size_t j, double lo, double hi) {
    bonds_[{std::min(i, j), std::max(i, j)}] = {lo, hi};
  }

  [[nodiscard]] std::string name() const { return "BondConstraint"; }

  [[nodiscard]] double
  compute_error(const coords_t &coords,
                std::span<const std::size_t> /*moved*/) const {
    double err = 0.0;
    for (auto &[key, bnd] : bonds_) {
      double d = distance(coords, key.first, key.second);
      if (d < bnd.lo)
        err += (bnd.lo - d);
      else if (d > bnd.hi)
        err += (d - bnd.hi);
    }
    return err;
  }

private:
  std::map<PairKey, Bound> bonds_;
};

} // namespace fullrmc
