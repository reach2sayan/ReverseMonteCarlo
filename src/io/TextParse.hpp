#pragma once
// Helpers shared by the structure/data readers and writers (private to src/io).
#include <RMC/core/Structure.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace RMC::io::detail {

// Trim ASCII whitespace from both ends, returning a sub-view (no allocation).
inline std::string_view trim(std::string_view sv) {
  constexpr std::string_view ws = " \t\r\n";
  const auto b = sv.find_first_not_of(ws);
  return b == std::string_view::npos
             ? std::string_view{}
             : sv.substr(b, sv.find_last_not_of(ws) - b + 1);
}

// Drop an inline `# ...` comment.
inline std::string_view strip_comment(std::string_view sv) {
  return sv.substr(0, sv.find('#'));
}

// v[i], or `fallback` for a per-atom field that is shorter than the structure.
template <class T, class U>
T at_or(const std::vector<T> &v, std::size_t i, U &&fallback) {
  return i < v.size() ? v[i] : T(std::forward<U>(fallback));
}

// Atom indices of one element, for writers that group atoms by species.
struct ElementGroup {
  std::string symbol;
  std::vector<std::size_t> atoms;
};

// Groups in first-appearance order; a missing/empty symbol becomes "X".
inline std::vector<ElementGroup> group_by_element(const AtomicStructure &s) {
  std::vector<ElementGroup> groups;
  for (std::size_t i = 0; i < s.size(); ++i) {
    std::string sym = at_or(s.elements, i, "X");
    if (sym.empty()) {
      sym = "X";
    }
    auto g = std::ranges::find(groups, sym, &ElementGroup::symbol);
    if (g == groups.end()) {
      g = groups.insert(groups.end(), ElementGroup{std::move(sym), {}});
    }
    g->atoms.push_back(i);
  }
  return groups;
}

// Collects atoms one at a time, then maps them into an AtomicStructure.
class StructureBuilder {
public:
  void add(const std::array<double, 3> &xyz, int atomic_number,
           std::string element, std::string name, std::string residue,
           std::size_t molecule) {
    xyz_.push_back(xyz);
    z_.push_back(atomic_number);
    s_.elements.push_back(std::move(element));
    s_.names.push_back(std::move(name));
    s_.residues.push_back(std::move(residue));
    s_.molecule_ids.push_back(molecule);
  }
  // Formats that carry only the species: name = residue = element, one molecule.
  void add(const std::array<double, 3> &xyz, int atomic_number,
           const std::string &element) {
    add(xyz, atomic_number, element, element, element, 1);
  }

  [[nodiscard]] std::size_t size() const noexcept { return xyz_.size(); }

  [[nodiscard]] AtomicStructure finish() && {
    const auto n = static_cast<Eigen::Index>(xyz_.size());
    s_.coordinates = Eigen::Map<const coords_t>(
        reinterpret_cast<const double *>(xyz_.data()), n, 3);
    s_.atomic_numbers = Eigen::Map<const ivec_t>(z_.data(), n);
    return std::move(s_);
  }

private:
  AtomicStructure s_;
  std::vector<std::array<double, 3>> xyz_;
  std::vector<int> z_;
};

} // namespace RMC::io::detail
