#pragma once
#include <algorithm>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace RMC {

// Unordered element pair, stored in canonical (sorted) order.
struct PairElemKey {
  std::string a, b;
  PairElemKey(std::string x, std::string y) : a(std::move(x)), b(std::move(y)) {
    if (b < a) {
      std::swap(a, b);
    }
  }
  auto operator<=>(const PairElemKey &) const = default;
};

// Dense ids for the distinct element symbols of a structure, assigned in sorted
// symbol order (so ids do not depend on atom order).
struct SpeciesIndex {
  std::vector<std::string> symbols; // sorted, distinct
  std::vector<std::uint8_t> id;     // per atom
  std::vector<std::size_t> count;   // atoms per species

  SpeciesIndex() = default;
  explicit SpeciesIndex(std::span<const std::string> elements)
      : symbols(elements.begin(), elements.end()) {
    std::ranges::sort(symbols);
    symbols.erase(std::ranges::unique(symbols).begin(), symbols.end());
    id = elements |
         std::views::transform([this](const std::string &e) { return *id_of(e); }) |
         std::ranges::to<std::vector>();
    count.assign(symbols.size(), 0);
    for (const std::uint8_t i : id) {
      ++count[i];
    }
  }

  [[nodiscard]] std::size_t size() const noexcept { return symbols.size(); }
  [[nodiscard]] std::optional<std::uint8_t> id_of(std::string_view s) const {
    const auto it = std::ranges::lower_bound(
        symbols, s, {}, [](const std::string &x) -> std::string_view { return x; });
    if (it == symbols.end() || *it != s) {
      return std::nullopt;
    }
    return static_cast<std::uint8_t>(it - symbols.begin());
  }
};

} // namespace RMC
