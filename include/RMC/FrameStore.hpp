#pragma once
#include <RMC/core/Structure.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace RMC {

// Frame storage with a uniform indexed interface (size/operator[]/primary);
// frame 0 is always "primary".

// A single frame on the heap at a STABLE address: a moved engine steals the
// unique_ptr, so references held by constraints/generators keep pointing at the
// same live structure.
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

// Frame storage on SingleFrameStore, so every frame keeps a stable heap address:
// a vector realloc during add() moves only the pointers, leaving structures put.
// References bound by constraints/generators survive both engine moves and add()
// growth — frames need not all be added before generator/constraint construction.
struct FrameStore {
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
