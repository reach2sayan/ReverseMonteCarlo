#include <RMC/constraints/BondConstraint.hpp>
#include <array>

namespace RMC {

double BondConstraint::compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
  return cache_.compute(
      bonds_, [](const BondItem &b) { return std::array{b.i, b.j}; },
      [this](const coords_t &c, const BondItem &b) {
        return range_violation(distance(c, b.i, b.j), b.lo, b.hi);
      },
      coords, moved);
}

} // namespace RMC
