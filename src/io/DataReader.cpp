#include <fullrmc/io/DataReader.hpp>
#include <boost/leaf/result.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>

namespace fullrmc::io {

Result<mat_t> read_xy_data(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) return boost::leaf::new_error(std::string{"Cannot open data file: " + path.string()});

    std::vector<std::array<double,2>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double x, y;
        if (!(ss >> x >> y)) continue;
        rows.push_back({x, y});
    }

    mat_t m(static_cast<Eigen::Index>(rows.size()), 2);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(rows.size()); ++i) {
        m(i, 0) = rows[static_cast<std::size_t>(i)][0];
        m(i, 1) = rows[static_cast<std::size_t>(i)][1];
    }
    return m;
}

} // namespace fullrmc::io
