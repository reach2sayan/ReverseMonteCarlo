#pragma once
#include <cstdint>

namespace RMC {

// Zero-temperature quench: accept a move iff it does not raise the total error
// beyond `tolerance` (default 0 → strict downhill).
class GreedySampler {
public:
  constexpr explicit GreedySampler(double tolerance = 0.0) noexcept
      : tolerance_{tolerance} {}
  [[nodiscard]] constexpr bool accept(double e_before, double e_after,
                                      std::uint64_t /*step*/,
                                      double /*u01*/) const noexcept {
    return e_after <= e_before + tolerance_;
  }
  [[nodiscard]] constexpr double tolerance() const noexcept {
    return tolerance_;
  }

private:
  double tolerance_;
};

} // namespace RMC
