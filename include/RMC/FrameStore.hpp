#pragma once
#include <RMC/core/Structure.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace RMC {

// Frame storage for the engines. Both stores present a uniform indexed
// interface (size(), operator[], primary()) so the shared step pipeline in
// EngineBase addresses frames the same way regardless of arity — frame 0 is
// always "primary".

// Single-frame storage. The structure lives on the heap at a STABLE address: a
// moved engine steals the unique_ptr, so any references held by its constraints
// / move generators (e.g. SQS's ClusterCorrelationConstraint, SpeciesSwap-
// Generator) keep pointing at the same live structure. This is the reason
// single-frame is NOT collapsed to a vector of size 1.
struct SingleFrameStore {
  std::unique_ptr<AtomicStructure> s_;

  SingleFrameStore() = default;
  explicit SingleFrameStore(AtomicStructure s)
      : s_(std::make_unique<AtomicStructure>(std::move(s))) {}

  [[nodiscard]] constexpr std::size_t size() const noexcept { return 1; }
  [[nodiscard]] constexpr AtomicStructure &operator[](std::size_t) noexcept {
    return *s_;
  }
  [[nodiscard]] constexpr const AtomicStructure &
  operator[](std::size_t) const noexcept {
    return *s_;
  }
  [[nodiscard]] constexpr AtomicStructure &primary() noexcept { return *s_; }
  [[nodiscard]] constexpr const AtomicStructure &primary() const noexcept {
    return *s_;
  }
};

// Multi-frame storage, built on SingleFrameStore so every frame inherits the
// stable-heap-address guarantee: each frame's AtomicStructure lives behind a
// unique_ptr, so a vector reallocation during add() merely moves the pointers
// (cheap) while the structures stay put. References bound by constraints / move
// generators therefore survive both engine moves AND add() growth, so frames
// need NOT all be added before generator/constraint construction.
struct MultiFrameStore {
  std::vector<SingleFrameStore> v_;

  void add(AtomicStructure s) { v_.emplace_back(std::move(s)); }
  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return v_.size();
  }
  [[nodiscard]] constexpr AtomicStructure &operator[](std::size_t i) noexcept {
    return v_[i].primary();
  }
  [[nodiscard]] constexpr const AtomicStructure &
  operator[](std::size_t i) const noexcept {
    return v_[i].primary();
  }
  [[nodiscard]] constexpr AtomicStructure &primary() noexcept {
    return v_[0].primary();
  }
  [[nodiscard]] constexpr const AtomicStructure &primary() const noexcept {
    return v_[0].primary();
  }
};

} // namespace RMC
