#pragma once
#include <RMC/core/TypeErasure.hpp>
#include <concepts>
#include <cstdint>
#include <memory>

namespace RMC {

class Sampler;

namespace detail {
struct SamplerToken {
private:
  constexpr SamplerToken() = default;
  friend class ::RMC::Sampler;
};
} // namespace detail

template <typename T>
concept CSampler = requires(T &s, detail::SamplerToken tok, double e_before,
                            double e_after, std::uint64_t step, double u01) {
  { s.accept(tok, e_before, e_after, step, u01) } -> std::convertible_to<bool>;
};

#define RMC_SAMPLER_METHODS                                                    \
  ((1, bool, accept,                                                           \
    (double e_before, double e_after, std::uint64_t step, double u01), 4,      \
    (e_before, e_after, step, u01), , , WITH_TOKEN))
RMC_DEFINE_ERASED_TYPE(Sampler, RMC_SAMPLER_METHODS)
#undef RMC_SAMPLER_METHODS

template <typename Derived> struct SamplerBase {};

} // namespace RMC
