#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>

#include <boost/leaf.hpp>

#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace RMC::analysis::detail {

// Shared precondition check for ADF / g(r): non-empty structure, elements match
// atom count, n_bins > 0, periodic box with positive volume. `fn` prefixes the
// error message; returns the cell volume V on success.
[[nodiscard]] FORCE_INLINE Result<double>
check_periodic_inputs(std::string_view fn, const coords_t &coords,
                      std::span<const std::string> elements, int n_bins,
                      const BoundaryConditions &bc) {
  const auto N = coords.rows();
  if (N == 0) {
    return boost::leaf::new_error(std::string{fn} + ": empty structure");
  } else if (n_bins <= 0) {
    return boost::leaf::new_error(std::string{fn} + ": need n_bins > 0");
  } else if (elements.size() != static_cast<std::size_t>(N)) {
    return boost::leaf::new_error(std::string{fn} +
                                  ": elements size does not match atom count");
  } else if (!std::holds_alternative<PeriodicBC>(bc)) {
    return boost::leaf::new_error(std::string{fn} +
                                  ": a periodic box is required");
  }
  const double V = bc_volume(bc);
  if (!(V > 0.0)) {
    return boost::leaf::new_error(std::string{fn} +
                                  ": box volume must be positive");
  }
  return V;
}

} // namespace RMC::analysis::detail
