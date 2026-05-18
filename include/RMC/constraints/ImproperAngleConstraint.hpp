#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace RMC {

// Improper dihedral: enforces planarity / chirality for quadruplets (i,j,k,l)
// where the angle is between plane(i,j,k) and plane(j,k,l).
//
// Incremental: on a single-atom move, only quads touching that atom are
// recomputed using cached per-quad error contributions.
class ImproperAngleConstraint : public ConstraintBase<ImproperAngleConstraint> {
public:
  struct Quad {
    std::size_t i, j, k, l;
    double lo, hi;
  };

  void add_improper(std::size_t i, std::size_t j, std::size_t k, std::size_t l,
                    double lo_rad, double hi_rad) {
    quads_.push_back({i, j, k, l, lo_rad, hi_rad});
    initialised_ = false;
  }

  [[nodiscard]] std::string name() const { return "ImproperAngleConstraint"; }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    if (!initialised_ || moved.empty()) {
      return full_recompute(coords);
    }
    for (std::size_t atom : moved) {
      auto it = atom_to_quads_.find(atom);
      if (it == atom_to_quads_.end()) {
        continue;
      }
      for (std::size_t qi : it->second) {
        double new_err = quad_error(coords, quads_[qi]);
        cached_total_ += new_err - quad_err_[qi];
        quad_err_[qi] = new_err;
      }
    }
    return cached_total_;
  }

private:
  double quad_error(const coords_t &c, const Quad &q) const noexcept {
    vec3_t b1 = c.row(q.j).transpose() - c.row(q.i).transpose();
    vec3_t b2 = c.row(q.k).transpose() - c.row(q.j).transpose();
    vec3_t b3 = c.row(q.l).transpose() - c.row(q.k).transpose();
    if (bc_) {
      b1 = bc_min_image(*bc_, b1);
      b2 = bc_min_image(*bc_, b2);
      b3 = bc_min_image(*bc_, b3);
    }
    vec3_t n1 = b1.cross(b2);
    vec3_t n2 = b2.cross(b3);
    double phi = std::atan2((n1.cross(n2)).dot(b2.normalized()), n1.dot(n2));
    if (phi < q.lo) {
      return q.lo - phi;
    }
    if (phi > q.hi) {
      return phi - q.hi;
    }
    return 0.0;
  }

  double full_recompute(const coords_t &coords) const {
    cached_total_ = 0.0;
    atom_to_quads_.clear();
    quad_err_.assign(quads_.size(), 0.0);
    for (std::size_t qi = 0; qi < quads_.size(); ++qi) {
      const auto &q = quads_[qi];
      double err = quad_error(coords, q);
      cached_total_ += err;
      quad_err_[qi] = err;
      atom_to_quads_[q.i].push_back(qi);
      atom_to_quads_[q.j].push_back(qi);
      atom_to_quads_[q.k].push_back(qi);
      atom_to_quads_[q.l].push_back(qi);
    }
    initialised_ = true;
    return cached_total_;
  }

  std::vector<Quad> quads_;

  mutable bool initialised_{false};
  mutable double cached_total_{0.0};
  mutable std::vector<double> quad_err_;
  mutable std::unordered_map<std::size_t, std::vector<std::size_t>>
      atom_to_quads_;
};

} // namespace RMC
