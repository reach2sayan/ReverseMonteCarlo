#pragma once
#include <cstdint>
#include <filesystem>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>

namespace RMC::io {

// Engine statistics snapshot saved/restored by checkpoints.
struct EngineStats {
  std::uint64_t steps_total{0};
  std::uint64_t steps_accepted{0};
  std::uint64_t steps_tried{0};
  double last_total_err{0.0};
};

// Save coordinates + stats to a binary archive (Boost.Serialization).
[[nodiscard]] Result<void> save_checkpoint(const AtomicStructure &s,
                                           const EngineStats &stats,
                                           const std::filesystem::path &path);

// Restore coordinates + stats from a previously saved checkpoint.
[[nodiscard]] Result<EngineStats>
load_checkpoint(AtomicStructure &s, const std::filesystem::path &path);

} // namespace RMC::io
