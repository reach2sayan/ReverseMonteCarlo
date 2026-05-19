#pragma once
#include <RMC/core/Types.hpp>
#include <RMC/generators/MoveGenerator.hpp>

#include <optional>
#include <string>
#include <vector>

namespace RMC {

struct Group {
  std::string name;
  std::vector<std::size_t> indices;
  std::optional<IMoveGenerator> generator;
  bool refine = true;

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return indices.size();
  }
  [[nodiscard]] constexpr bool empty() const noexcept {
    return indices.empty();
  }
  [[nodiscard]] constexpr std::span<const std::size_t> span() const noexcept {
    return indices;
  }
};

} // namespace RMC
