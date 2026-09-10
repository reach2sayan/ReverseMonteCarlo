#include "TextParse.hpp"

#include <RMC/io/DataReader.hpp>
#include <boost/leaf/result.hpp>
#include <boost/parser/parser.hpp>
#include <fstream>
#include <string>
#include <vector>

namespace RMC::io {

namespace {

namespace bp = boost::parser;

// Numeric table: the numeric prefix of each line (whitespace or comma
// separated; '#' starts a comment) as an N × ncol matrix. With ncol fixed,
// extra columns are dropped and shorter rows skipped; with ncol == 0 the first
// row sets the width and a ragged row is an error.
Result<mat_t> read_table(const std::filesystem::path &path, std::size_t ncol) {
  std::ifstream f(path);
  if (!f) {
    return boost::leaf::new_error("Cannot open data file: " + path.string());
  }
  const bool fixed = ncol != 0;
  std::vector<double> values, row;
  for (std::string line; std::getline(f, line);) {
    const std::string_view content = detail::strip_comment(line);
    auto it = content.begin();
    row.clear();
    bp::prefix_parse(it, content.end(), *bp::double_, bp::char_(" \t,\r"), row);
    if (row.empty() || (fixed && row.size() < ncol)) {
      continue;
    }
    if (ncol == 0) {
      ncol = row.size();
    } else if (!fixed && row.size() != ncol) {
      return boost::leaf::new_error("read_columns: ragged rows in " +
                                    path.string());
    }
    values.insert(values.end(), row.begin(),
                  row.begin() + static_cast<std::ptrdiff_t>(ncol));
  }
  if (ncol == 0) {
    return boost::leaf::new_error("read_columns: no numeric data in " +
                                  path.string());
  }
  using RowMajor =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  return mat_t(Eigen::Map<const RowMajor>(
      values.data(), static_cast<Eigen::Index>(values.size() / ncol),
      static_cast<Eigen::Index>(ncol)));
}

} // namespace

Result<mat_t> read_xy_data(const std::filesystem::path &path) {
  return read_table(path, 2);
}

Result<mat_t> read_columns(const std::filesystem::path &path) {
  return read_table(path, 0);
}

} // namespace RMC::io
