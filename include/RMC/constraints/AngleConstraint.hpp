#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace RMC {

// Enforces bond-angle bounds for atom triplets (i–j–k), angle at j.
//
// Incremental: on a single-atom move, only triplets touching that atom are
// recomputed. All others use their cached per-triplet error contribution.
class AngleConstraint : public ConstraintBase<AngleConstraint> {
public:
  struct Triplet {
    std::size_t i, j, k;
    double lo, hi;
  };

  void add_angle(std::size_t i, std::size_t j, std::size_t k, double lo_rad,
                 double hi_rad) {
    triplets_.push_back({i, j, k, lo_rad, hi_rad});
    initialised_ = false;
  }

  [[nodiscard]] std::string name() const { return "AngleConstraint"; }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    if (!initialised_ || moved.empty())
      return full_recompute(coords);
    for (std::size_t atom : moved) {
      auto it = atom_to_triplets_.find(atom);
      if (it == atom_to_triplets_.end())
        continue;
      for (std::size_t ti : it->second) {
        double new_err = triplet_error(coords, triplets_[ti]);
        cached_total_ += new_err - triplet_err_[ti];
        triplet_err_[ti] = new_err;
      }
    }
    return cached_total_;
  }

private:
  double triplet_error(const coords_t &coords, const Triplet &t) const {
    vec3_t v1 = (coords.row(t.i) - coords.row(t.j)).transpose();
    vec3_t v2 = (coords.row(t.k) - coords.row(t.j)).transpose();
    if (bc_) {
      v1 = bc_min_image(*bc_, v1);
      v2 = bc_min_image(*bc_, v2);
    }
    double cos_a = v1.dot(v2) / (v1.norm() * v2.norm() + 1e-30);
    double angle = std::acos(std::clamp(cos_a, -1.0, 1.0));
    if (angle < t.lo)
      return t.lo - angle;
    if (angle > t.hi)
      return angle - t.hi;
    return 0.0;
  }

  double full_recompute(const coords_t &coords) const {
    cached_total_ = 0.0;
    atom_to_triplets_.clear();
    triplet_err_.assign(triplets_.size(), 0.0);
    for (std::size_t ti = 0; ti < triplets_.size(); ++ti) {
      const auto &t = triplets_[ti];
      double err = triplet_error(coords, t);
      cached_total_ += err;
      triplet_err_[ti] = err;
      atom_to_triplets_[t.i].push_back(ti);
      atom_to_triplets_[t.j].push_back(ti);
      atom_to_triplets_[t.k].push_back(ti);
    }
    initialised_ = true;
    return cached_total_;
  }

  std::vector<Triplet> triplets_;

  mutable bool initialised_{false};
  mutable double cached_total_{0.0};
  mutable std::vector<double> triplet_err_;
  mutable std::unordered_map<std::size_t, std::vector<std::size_t>>
      atom_to_triplets_;
};

} // namespace RMC
