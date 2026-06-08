#pragma once
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <utility>

namespace RMC::callbacks {

// Dumps the computed vs experimental histogram to a three-column CSV on every
// invocation. The histogram data is accessed through a user-supplied getter
// that captures the relevant constraint(s) at construction time — the only
// engine state allowed beyond the guaranteed callback params.
//
// Usage:
//   auto &c = my_pdf_constraint;
//   engine.set_step_callback(
//       HistogramCallback{
//           .axis   = c.r_axis(),   // or any pre-built vec_t of r/Q values
//           .getter = [&c]{ return std::make_pair(c.computed_G(),
//                                                 c.experimental_data()); },
//           .dir    = "histograms/"
//       }, 500);
struct HistogramCallback {
  // Returns (computed, experimental) as a pair of column vectors.
  // Capture whatever constraint reference you need at construction time.
  using Getter = std::function<std::pair<vec_t, vec_t>()>;
  vec_t axis;    // r or Q axis values (captured at construction)
  Getter getter; // captures constraint ref(s) at construction
  std::filesystem::path dir;

  void operator()(std::uint64_t step, std::uint64_t acc, std::uint64_t tried,
                  double chi2, const AtomicStructure &s) const;
};

} // namespace RMC::callbacks
