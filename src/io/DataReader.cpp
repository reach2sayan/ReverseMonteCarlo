#include <RMC/io/DataReader.hpp>
#include <algorithm>
#include <array>
#include <boost/leaf/result.hpp>
#include <boost/parser/parser.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace RMC::io {

namespace bp = boost::parser;

Result<mat_t> read_xy_data(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f) {
    return boost::leaf::new_error(
        std::string{"Cannot open data file: " + path.string()});
  }
  // Two reals per row; columns separated by whitespace OR commas. Any trailing
  // columns are ignored (prefix_parse stops after the second number), matching
  // the historic `ss >> x >> y` leniency while honoring the documented comma
  // support.
  const auto row_p = bp::double_ >> bp::double_;
  const auto sep = bp::char_(" \t,\r");

  std::vector<std::array<double, 2>> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    auto it = line.begin();
    if (const auto xy = bp::prefix_parse(it, line.end(), row_p, sep)) {
      const auto &[x, y] = *xy;
      rows.push_back({x, y});
    }
    // Lines with fewer than two numbers are silently skipped, as before.
  }

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>
      rows_map(reinterpret_cast<const double *>(rows.data()),
               static_cast<Eigen::Index>(rows.size()), 2);

  mat_t m = rows_map;
  return m;
}

Result<mat_t> read_columns(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f) {
    return boost::leaf::new_error(
        std::string{"Cannot open data file: " + path.string()});
  }

  std::vector<std::vector<double>> rows;
  std::size_t ncol = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (const auto h = line.find('#'); h != std::string::npos) {
      line.resize(h); // strip inline/whole-line comments
    }
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream ss{line};
    std::vector<double> row;
    for (double v; ss >> v;) {
      row.push_back(v);
    }
    if (row.empty()) {
      continue;
    }
    if (ncol == 0) {
      ncol = row.size();
    } else if (row.size() != ncol) {
      return boost::leaf::new_error(
          std::string{"read_columns: ragged rows in " + path.string()});
    }
    rows.push_back(std::move(row));
  }
  if (rows.empty()) {
    return boost::leaf::new_error(
        std::string{"read_columns: no numeric data in " + path.string()});
  }

  mat_t m(static_cast<Eigen::Index>(rows.size()),
          static_cast<Eigen::Index>(ncol));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < ncol; ++j) {
      m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          rows[i][j];
    }
  }
  return m;
}

} // namespace RMC::io
