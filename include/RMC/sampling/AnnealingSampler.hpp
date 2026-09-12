#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace RMC {

// Simulated annealing: Metropolis acceptance with a geometric cooling schedule
//
//   T(step) = max(t_min, t0 · cooling^(step / interval))
//
// so the walker accepts many uphill moves early (high T) and progressively
// fewer as it cools — the mechanism ATAT's mcsqs relies on to escape ordered
// local minima during SQS search, which RMC's previous greedy-only acceptance
// could not. `cooling` ∈ (0,1); `t_min` floors T to avoid division by zero.
class AnnealingSampler {
public:
  struct Schedule {
    double t0 = 1.0;                // initial temperature
    double cooling = 0.9;           // geometric factor per interval (<1 cools)
    std::uint64_t interval = 10000; // steps between successive cooling steps
    double t_min = 1e-6;            // temperature floor
  };

  constexpr AnnealingSampler() noexcept = default;
  constexpr explicit AnnealingSampler(Schedule s) noexcept : s_{std::move(s)} {}

  [[nodiscard]] double temperature(std::uint64_t step) const noexcept {
    const std::uint64_t completed_cools = step / std::max<std::uint64_t>(s_.interval, 1);
    return std::max(s_.t_min,
                    s_.t0 * std::pow(s_.cooling, static_cast<double>(completed_cools)));
  }

  [[nodiscard]] bool accept(double e_before, double e_after, std::uint64_t step,
                            double u01) const noexcept {
    const double dE = e_after - e_before;
    return dE <= 0.0 || u01 < std::exp(-dE / temperature(step));
  }

private:
  Schedule s_{};
};

} // namespace RMC
