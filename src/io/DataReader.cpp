#include <RMC/io/DataReader.hpp>
#include <array>
#include <boost/leaf/result.hpp>
#include <boost/parser/parser.hpp>
#include <fstream>
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

} // namespace RMC::io
