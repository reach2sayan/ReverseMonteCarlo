#include <array>
#include <boost/leaf/result.hpp>
#include <fstream>
#include <RMC/io/DataReader.hpp>
#include <sstream>
#include <vector>

namespace RMC::io {

Result<mat_t> read_xy_data(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f) {
    return boost::leaf::new_error(
        std::string{"Cannot open data file: " + path.string()});
  }
  std::vector<std::array<double, 2>> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ss(line);
    double x, y;
    if (!(ss >> x >> y)) {
      continue;
    }
    rows.push_back({x, y});
  }

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>
      rows_map(reinterpret_cast<const double *>(rows.data()),
               static_cast<Eigen::Index>(rows.size()), 2);

  mat_t m = rows_map;
  return m;
}

} // namespace RMC::io
