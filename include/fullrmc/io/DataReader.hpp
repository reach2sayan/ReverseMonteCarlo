#pragma once
#include <fullrmc/core/Types.hpp>
#include <filesystem>
namespace fullrmc::io {

// Read a whitespace/comma-separated two-column data file (e.g. r vs G(r),
// Q vs S(Q)).  Lines beginning with '#' are treated as comments.
// Returns an N×2 matrix where column 0 = x, column 1 = y.
[[nodiscard]] Result<mat_t>
read_xy_data(const std::filesystem::path& path);

} // namespace fullrmc::io
