#include <RMC/callbacks/PDBSnapshotCallback.hpp>
#include <RMC/io/PdbReader.hpp>

#include <spdlog/spdlog.h>

#include <format>

namespace RMC::callbacks {

void PDBSnapshotCallback::operator()(std::uint64_t step, std::uint64_t /*acc*/,
                                     std::uint64_t /*tried*/, double /*chi2*/,
                                     const AtomicStructure &s) const {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  auto path = dir / std::format("step_{:010d}.pdb", step);
  if (auto r = io::write_pdb(s, path); !r) {
    spdlog::warn("PDBSnapshotCallback: write failed at step {}", step);
  }
}

} // namespace RMC::callbacks
