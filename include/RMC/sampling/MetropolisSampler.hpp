#pragma once
#include <RMC/sampling/Sampler.hpp>
#include <cmath>

namespace RMC {

// Classical Metropolis acceptance at a fixed temperature T:
//   accept if ΔE ≤ 0, else accept with probability exp(-ΔE / T).
//
// This is the canonical Reverse Monte Carlo criterion. For data fitting, fold
// the experimental uncertainty σ into each constraint's χ² and use T = 2 to
// recover the McGreevy–Pusztai exp(-Δχ²/2) rule; T then plays the role of the
// statistical "temperature" that lets the walker sample the data-consistent
// ensemble rather than greedily quenching into a single best-fit.
class MetropolisSampler : public SamplerBase<MetropolisSampler> {
public:
  constexpr explicit MetropolisSampler(double temperature) noexcept
      : t_(temperature) {}
  [[nodiscard]] bool accept(Sampler::Token, double e_before, double e_after,
                            std::uint64_t /*step*/, double u01) const noexcept {
    const double dE = e_after - e_before;
    if (dE <= 0.0) {
      return true;
    }
    if (t_ <= 0.0) {
      return false; // T → 0 degenerates to a pure greedy quench.
    }
    return u01 < std::exp(-dE / t_);
  }

  [[nodiscard]] constexpr double temperature() const noexcept { return t_; }
private:
  double t_;
};

} // namespace RMC
