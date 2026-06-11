#pragma once
#include <RMC/core/Types.hpp>
#include <boost/container/flat_set.hpp>
#include <cstddef>
#include <string>

namespace RMC::analysis {

// One species' contribution to a configuration. Pairing a symbol with its
// atom count in a single value means the two can never drift out of sync the
// way two parallel `species`/`counts` vectors could. Ordered/deduplicated by
// symbol so it can live in a flat_set keyed on the element.
struct SpeciesCount {
  std::string symbol;    // element symbol, e.g. "Zr"
  std::size_t count = 0; // number of atoms of this species

  friend bool operator<(const SpeciesCount &a, const SpeciesCount &b) {
    return a.symbol < b.symbol;
  }
};

// The distinct elements of a configuration with their atom counts. A flat_set
// makes "distinct, sorted by symbol" an invariant of the type rather than
// something each producer has to maintain by hand.
using Composition = boost::container::flat_set<SpeciesCount>;

// A named curve sharing its parent result's x-grid: one partial g(r) or ADF,
// tagged with the element pair/triplet it belongs to. The label travels with
// the data instead of living in a separate, index-aligned label vector.
struct LabeledCurve {
  std::string label; // e.g. "Zr-Cu" (pair) or "Zr-Cu-Cu" (triplet)
  vec_t values;      // length n_bins, aligned with the parent grid
};

} // namespace RMC::analysis
