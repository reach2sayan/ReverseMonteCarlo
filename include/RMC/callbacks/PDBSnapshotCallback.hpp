#pragma once
#include <RMC/core/Structure.hpp>
#include <RMC/io/PdbReader.hpp>

#include <boost/log/trivial.hpp>

#include <cstdint>
#include <filesystem>
#include <format>

namespace RMC::callbacks {

// Writes the current structure to a zero-padded PDB file on every invocation.
// Files are named <dir>/step_NNNNNNNNNN.pdb so lexicographic order matches
// step order. The output directory is created on first use if absent.
struct PDBSnapshotCallback {
  std::filesystem::path dir;

  void operator()(std::uint64_t step, std::uint64_t /*acc*/,
                  std::uint64_t /*tried*/, double /*chi2*/,
                  const AtomicStructure &s) const {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    auto path = dir / std::format("step_{:010d}.pdb", step);
    if (auto r = io::write_pdb(s, path); !r) {
      BOOST_LOG_TRIVIAL(warning)
          << "PDBSnapshotCallback: write failed at step " << step;
    }
  }
};

} // namespace RMC::callbacks
