#pragma once
#include <RMC/sampling/AnnealingSampler.hpp>
#include <RMC/sampling/GreedySampler.hpp>
#include <RMC/sampling/MetropolisSampler.hpp>
#include <cstdint>
#include <variant>

namespace RMC {

// The move-acceptance policy: one of the samplers above, dispatched by value.
struct Sampler
    : std::variant<GreedySampler, MetropolisSampler, AnnealingSampler> {
  using variant::variant;
  [[nodiscard]] bool accept(double e_before, double e_after, std::uint64_t step,
                            double u01) const {
    return std::visit(
        [&](const auto &s) { return s.accept(e_before, e_after, step, u01); },
        *this);
  }
};

} // namespace RMC
