#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>

#include <boost/leaf.hpp>

#include <algorithm>
#include <array>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace RMC::analysis::detail {

// Shared precondition check for ADF / g(r): non-empty structure, n_bins > 0,
// elements match atom count, periodic box with positive volume. The first
// failed check becomes the error, prefixed by `fn`; returns the cell volume V.
[[nodiscard]] inline Result<double>
check_periodic_inputs(std::string_view fn, const coords_t &coords,
                      std::span<const std::string> elements, int n_bins,
                      const BoundaryConditions &bc) {
  const std::array<std::pair<bool, std::string_view>, 5> checks{{
      {coords.rows() == 0, "empty structure"},
      {n_bins <= 0, "need n_bins > 0"},
      {elements.size() != static_cast<std::size_t>(coords.rows()),
       "elements size does not match atom count"},
      {!bc.periodic(), "a periodic box is required"},
      {!(bc.volume() > 0.0), "box volume must be positive"},
  }};
  if (const auto bad =
          std::ranges::find_if(checks, [](const auto &c) { return c.first; });
      bad != checks.end()) {
    return boost::leaf::new_error(std::format("{}: {}", fn, bad->second));
  }
  return bc.volume();
}

} // namespace RMC::analysis::detail
