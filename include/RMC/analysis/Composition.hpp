#pragma once
#include <RMC/core/SpeciesIndex.hpp>
#include <RMC/core/Types.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/leaf/error.hpp>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <ranges>
#include <span>
#include <string>
#include <string_view>

namespace RMC::analysis {

struct SpeciesCount {
  std::string symbol;
  std::size_t count = 0;
  friend bool operator<(const SpeciesCount &a, const SpeciesCount &b) {
    return a.symbol < b.symbol;
  }
};

using Composition = boost::container::flat_set<SpeciesCount>;
struct LabeledCurve {
  std::string label; // e.g. "Zr-Cu" (pair) or "Zr-Cu-Cu" (triplet)
  vec_t values;      // length n_bins, aligned with the parent grid
};

// Species symbols of `sp` with their atom counts.
[[nodiscard]] inline Composition composition_of(const SpeciesIndex &sp) {
  Composition out;
  for (const auto &[symbol, count] : std::views::zip(sp.symbols, sp.count)) {
    out.insert({symbol, count});
  }
  return out;
}

// Write `x` and each curve as a commented whitespace table (the format
// read_xy_data consumes): "# <title>, species=A(n),…", a column-name line
// ("# <x_name>  <prefix><label>…"), then one row per grid point.
[[nodiscard]] inline Result<void>
write_curves(const std::filesystem::path &path, std::string_view title,
             const Composition &species, std::string_view x_name,
             const vec_t &x, std::span<const LabeledCurve> curves,
             std::string_view prefix) {
  std::ofstream f(path);
  if (!f) {
    return boost::leaf::new_error(std::format("Cannot write {}", path.string()));
  }
  f << "# " << title << ", species=";
  for (const auto &[i, s] : species | std::views::enumerate) {
    f << (i ? "," : "") << s.symbol << "(" << s.count << ")";
  }
  f << "\n# " << x_name;
  for (const auto &c : curves) {
    f << "  " << prefix << c.label;
  }
  f << "\n";
  for (Eigen::Index k = 0; k < x.size(); ++k) {
    f << std::format("{:.6f}", x(k));
    for (const auto &c : curves) {
      f << std::format(" {:.8f}", c.values(k));
    }
    f << "\n";
  }
  return {};
}

} // namespace RMC::analysis
