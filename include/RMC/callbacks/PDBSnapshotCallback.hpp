#pragma once
#include <RMC/core/Structure.hpp>

#include <cstdint>
#include <filesystem>

namespace RMC::callbacks {

// Writes the current structure to a zero-padded PDB file on every invocation.
// Files are named <dir>/step_NNNNNNNNNN.pdb so lexicographic order matches
// step order. The output directory is created on first use if absent.
struct PDBSnapshotCallback {
  std::filesystem::path dir;

  void operator()(std::uint64_t step, std::uint64_t acc, std::uint64_t tried,
                  double chi2, const AtomicStructure &s) const;
};

} // namespace RMC::callbacks
