#include <RMC/io/Checkpoint.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/leaf/result.hpp>
#include <boost/serialization/array_wrapper.hpp>
#include <format>
#include <fstream>

// Teach Boost.Serialization how to handle Eigen dense matrices.
namespace boost::serialization {
template <class Archive, class Scalar, int Rows, int Cols, int Options,
          int MaxRows, int MaxCols>
void serialize(Archive &ar,
               Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &m,
               const unsigned int /*version*/) {
  Eigen::Index rows = m.rows(), cols = m.cols();
  ar & rows & cols;
  if constexpr (Archive::is_loading::value) {
    m.resize(rows, cols);
  }
  if (m.size() > 0) {
    ar &boost::serialization::make_array(m.data(),
                                         static_cast<std::size_t>(m.size()));
  }
}

} // namespace boost::serialization

namespace RMC::io {

namespace {

// The checkpoint layout, shared by save (oarchive) and load (iarchive).
template <class Archive, class Coords, class Stats>
void transfer(Archive &ar, Coords &coords, Stats &stats) {
  ar & coords & stats.steps_total & stats.steps_accepted & stats.steps_tried &
      stats.last_total_err;
}

// Opens `path` as a binary Archive over Stream and runs body(ar); open
// failures and archive exceptions become leaf errors. `verb` is read/write.
template <class Archive, class Stream, class Body>
Result<void> with_archive(const std::filesystem::path &path,
                          std::string_view verb, Body &&body) {
  try {
    Stream f(path, std::ios::binary);
    if (!f) {
      return boost::leaf::new_error(
          std::format("Cannot {} checkpoint: {}", verb, path.string()));
    }
    Archive ar(f);
    body(ar);
    return {};
  } catch (const std::exception &e) {
    return boost::leaf::new_error(
        std::format("Checkpoint {} error: {}", verb, e.what()));
  }
}

} // namespace

Result<void> save_checkpoint(const AtomicStructure &s, const EngineStats &stats,
                             const std::filesystem::path &path) {
  return with_archive<boost::archive::binary_oarchive, std::ofstream>(
      path, "write", [&](auto &ar) { transfer(ar, s.coordinates, stats); });
}

Result<EngineStats> load_checkpoint(AtomicStructure &s,
                                    const std::filesystem::path &path) {
  EngineStats stats;
  BOOST_LEAF_CHECK((with_archive<boost::archive::binary_iarchive, std::ifstream>(
      path, "read", [&](auto &ar) { transfer(ar, s.coordinates, stats); })));
  return stats;
}

} // namespace RMC::io
