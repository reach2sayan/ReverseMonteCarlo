#pragma once
#include <fullrmc/core/Types.hpp>
#include <fullrmc/core/Structure.hpp>
#include <filesystem>
#include <string>

namespace fullrmc::io {

// Parses ATOM and HETATM records from a PDB file.
// Returns an AtomicStructure or an error string.
[[nodiscard]] Result<AtomicStructure>
read_pdb(const std::filesystem::path& path);

// Write coordinates back to PDB format (useful for checkpointing).
[[nodiscard]] Result<void>
write_pdb(const AtomicStructure& s, const std::filesystem::path& path);

} // namespace fullrmc::io
