#pragma once
#include <RMC/core/Types.hpp>
#include <filesystem>
namespace RMC::io {

// Read a whitespace/comma-separated two-column data file (e.g. r vs G(r),
// Q vs S(Q)).  Lines beginning with '#' are treated as comments.
// Returns an N×2 matrix where column 0 = x, column 1 = y.
[[nodiscard]] Result<mat_t> read_xy_data(const std::filesystem::path &path);

// Read a whitespace/comma-separated numeric table with an arbitrary but fixed
// number of columns (e.g. an ADF target: angle + one column per triplet). Lines
// beginning with '#' are comments. Returns an N×M matrix; fails on ragged rows.
[[nodiscard]] Result<mat_t> read_columns(const std::filesystem::path &path);

} // namespace RMC::io
