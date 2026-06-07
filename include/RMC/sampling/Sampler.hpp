#pragma once
#include <RMC/core/TypeErasure.hpp>
#include <concepts>
#include <cstdint>
#include <memory>

namespace RMC {

class Sampler; // forward for friend declaration

namespace detail {
struct SamplerToken {
private:
  constexpr SamplerToken() = default;
  friend class ::RMC::Sampler;
};
} // namespace detail

// Satisfied by any type that implements the passkey-gated acceptance policy.
//
// A sampler decides whether a proposed move is accepted, given the total error
// (summed over SOFT constraints) before and after the move. `step` is the
// global MC step index (so annealing schedules can cool over time) and `u01` is
// a fresh uniform draw in [0,1) supplied by the engine — the policy owns no RNG
// itself, which keeps it trivially copyable for the island-model ensemble
// factory.
template <typename T>
concept CSampler = requires(T &s, detail::SamplerToken tok, double e_before,
                            double e_after, std::uint64_t step, double u01) {
  { s.accept(tok, e_before, e_after, step, u01) } -> std::convertible_to<bool>;
};

// Type-erased acceptance policy (mirrors GroupSelector / MoveGenerator).
#define RMC_SAMPLER_METHODS                                                    \
  ((1, bool, accept,                                                           \
    (double e_before, double e_after, std::uint64_t step, double u01), 4,      \
    (e_before, e_after, step, u01), , , WITH_TOKEN))
RMC_DEFINE_ERASED_TYPE(Sampler, RMC_SAMPLER_METHODS)
#undef RMC_SAMPLER_METHODS

template <typename Derived> struct SamplerBase {};

} // namespace RMC
