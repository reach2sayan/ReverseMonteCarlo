#pragma once
#include <RMC/core/Types.hpp>
#include <boost/container/flat_set.hpp>
#include <cstddef>
#include <string>

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

} // namespace RMC::analysis
