#pragma once
#include <fullrmc/core/Types.hpp>
#include <fullrmc/generators/MoveGenerator.hpp>

#include <optional>
#include <string>
#include <vector>

namespace fullrmc {

struct Group {
  std::string name;
  std::vector<std::size_t> indices;
  std::optional<IMoveGenerator> generator;
  const bool refine = true;

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

} // namespace fullrmc
