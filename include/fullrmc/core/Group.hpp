#pragma once
#include <fullrmc/core/Types.hpp>
#include <memory>
#include <string>
#include <vector>

namespace fullrmc {

struct IMoveGenerator;
struct Group {
  std::string name;
  std::vector<std::size_t> indices; // atom indices belonging to this group
  std::shared_ptr<IMoveGenerator> generator; // how this group moves
  bool refine = true;                        // participate in refinement?
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
