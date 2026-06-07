#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalCache.hpp>
#include <vector>

namespace RMC {

// Enforces dihedral-angle bounds for atom quadruplets (i–j–k–l).
//
// Incremental: on a single-atom move, only quads touching that atom are
// recomputed via ItemCache. All others retain their cached contribution.
class DihedralAngleConstraint
    : public RigidConstraintBase<DihedralAngleConstraint> {
public:
  struct Quad {
    std::size_t i, j, k, l;
    double lo, hi;
  };

  void add_dihedral(std::size_t i, std::size_t j, std::size_t k, std::size_t l,
                    double lo_rad, double hi_rad) {
    quads_.push_back({i, j, k, l, lo_rad, hi_rad});
    cache_.invalidate();
  }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "DihedralAngleConstraint";
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

private:
  double quad_error(const coords_t &c, const Quad &q) const noexcept;

  std::vector<Quad> quads_;
  mutable ItemCache<Quad> cache_;
};

static_assert(CConstraint<DihedralAngleConstraint>,
              "DihedralAngleConstraint must satisfy the CConstraint concept");

} // namespace RMC
