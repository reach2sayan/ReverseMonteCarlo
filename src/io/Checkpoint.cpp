#include <fullrmc/io/Checkpoint.hpp>
#include <boost/leaf/result.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <fstream>
#include <sstream>

// Teach Boost.Serialization how to handle Eigen dense matrices.
namespace boost::serialization {

template<class Archive, class Scalar, int Rows, int Cols, int Options,
         int MaxRows, int MaxCols>
void serialize(Archive& ar,
               Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& m,
               const unsigned int /*version*/) {
    Eigen::Index rows = m.rows(), cols = m.cols();
    ar & rows & cols;
    if constexpr (Archive::is_loading::value)
        m.resize(rows, cols);
    if (m.size() > 0)
        ar & boost::serialization::make_array(m.data(),
                                               static_cast<std::size_t>(m.size()));
}

} // namespace boost::serialization

namespace fullrmc::io {

Result<void> save_checkpoint(const AtomicStructure& s,
                               const EngineStats& stats,
                               const std::filesystem::path& path) {
    try {
        std::ofstream f(path, std::ios::binary);
        if (!f) return boost::leaf::new_error(std::string{"Cannot write checkpoint: " + path.string()});
        boost::archive::binary_oarchive ar(f);
        ar & s.coordinates;
        ar & stats.steps_total;
        ar & stats.steps_accepted;
        ar & stats.steps_tried;
        ar & stats.last_total_err;
        return {};
    } catch (const std::exception& e) {
        return boost::leaf::new_error(std::string{"Checkpoint write error: "} + e.what());
    }
}

Result<EngineStats> load_checkpoint(AtomicStructure& s,
                                     const std::filesystem::path& path) {
    try {
        std::ifstream f(path, std::ios::binary);
        if (!f) return boost::leaf::new_error(std::string{"Cannot read checkpoint: " + path.string()});
        boost::archive::binary_iarchive ar(f);
        ar & s.coordinates;
        EngineStats stats;
        ar & stats.steps_total;
        ar & stats.steps_accepted;
        ar & stats.steps_tried;
        ar & stats.last_total_err;
        return stats;
    } catch (const std::exception& e) {
        return boost::leaf::new_error(std::string{"Checkpoint read error: "} + e.what());
    }
}

} // namespace fullrmc::io
