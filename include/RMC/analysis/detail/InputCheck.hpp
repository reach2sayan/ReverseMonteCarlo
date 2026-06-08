#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>

#include <boost/leaf.hpp>

#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace RMC::analysis::detail {

// Shared precondition check for the in-memory ADF / g(r) entry points: a
// non-empty structure whose element labels match the atom count, a positive
// bin count, and a periodic box with positive volume (both normalisations need
// a finite cell volume). `fn` prefixes the error message so callers keep their
// distinct diagnostics. Returns the cell volume V on success for reuse.
[[nodiscard]] inline Result<double>
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
