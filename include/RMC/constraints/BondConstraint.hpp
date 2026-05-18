#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <algorithm>
#include <boost/container/flat_map.hpp>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

namespace RMC {

// Enforces bond-length bounds for explicit atom-index pairs.
// std_err = Σ violations where violation > 0 means bond is outside [lo, hi].
//
// Incremental: on a single-atom move, only bonds touching that atom are
// recomputed. All others use their cached per-bond error contribution.
class BondConstraint : public ConstraintBase<BondConstraint> {
public:
  struct Bound {
    double lo, hi;
  };
  using PairKey = std::pair<std::size_t, std::size_t>;

  void add_bond(std::size_t i, std::size_t j, double lo, double hi) {
    auto key = PairKey{std::min(i, j), std::max(i, j)};
    bonds_[key] = {lo, hi};
    initialised_ = false; // invalidate cache
  }

  [[nodiscard]] std::string name() const { return "BondConstraint"; }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    if (!initialised_ || moved.empty()) {
      return full_recompute(coords);
    }
    for (std::size_t atom : moved) {
      auto it = atom_to_bonds_.find(atom);
      if (it == atom_to_bonds_.end())
        continue;
      for (const PairKey &key : it->second) {
        double d = distance(coords, key.first, key.second);
        const auto &bnd = bonds_.at(key);
        double old_err = bond_err_.at(key);
        double new_err = 0.0;
        if (d < bnd.lo)
          new_err = bnd.lo - d;
        else if (d > bnd.hi)
          new_err = d - bnd.hi;
        cached_total_ += new_err - old_err;
        bond_err_.at(key) = new_err;
      }
    }
    return cached_total_;
  }

private:
  double full_recompute(const coords_t &coords) const {
    cached_total_ = 0.0;
    atom_to_bonds_.clear();
    bond_err_.clear();
    for (auto &[key, bnd] : bonds_) {
      double d = distance(coords, key.first, key.second);
      double err = 0.0;
      if (d < bnd.lo)
        err = bnd.lo - d;
      else if (d > bnd.hi)
        err = d - bnd.hi;
      cached_total_ += err;
      bond_err_[key] = err;
      atom_to_bonds_[key.first].push_back(key);
      atom_to_bonds_[key.second].push_back(key);
    }
    initialised_ = true;
    return cached_total_;
  }

  boost::container::flat_map<PairKey, Bound> bonds_;

  mutable bool initialised_{false};
  mutable double cached_total_{0.0};
  mutable std::unordered_map<std::size_t, std::vector<PairKey>> atom_to_bonds_;
  mutable boost::container::flat_map<PairKey, double> bond_err_;
};

} // namespace RMC
