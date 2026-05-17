#pragma once
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
#include <filesystem>
#include <string>

namespace RMC::io {

// Parses ATOM and HETATM records from a PDB file.
// Returns an AtomicStructure or an error string.
[[nodiscard]] Result<AtomicStructure>
read_pdb(const std::filesystem::path &path);

// Write coordinates back to PDB format (useful for checkpointing).
[[nodiscard]] Result<void> write_pdb(const AtomicStructure &s,
                                     const std::filesystem::path &path);

} // namespace RMC::io
