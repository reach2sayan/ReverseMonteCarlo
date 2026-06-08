#include <RMC/callbacks/HistogramCallback.hpp>

#include <boost/log/trivial.hpp>

#include <algorithm>
#include <format>
#include <fstream>

namespace RMC::callbacks {

void HistogramCallback::operator()(std::uint64_t step, std::uint64_t /*acc*/,
                                   std::uint64_t /*tried*/, double /*chi2*/,
                                   const AtomicStructure & /*s*/) const {
  if (!getter) {
    return;
  }
  auto [computed, experimental] = getter();
  const Eigen::Index n =
      std::min({axis.size(), computed.size(), experimental.size()});
  if (n == 0) {
    return;
  }

  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  auto path = dir / std::format("hist_{:010d}.csv", step);
  std::ofstream f(path);
  if (!f) {
    BOOST_LOG_TRIVIAL(warning) << "HistogramCallback: could not open " << path;
    return;
  }
  f << "axis,computed,experimental\n";
  for (const auto &[ax, comp, exp] :
       std::views::zip(axis, computed, experimental)) {
    f << std::format("{:.6g},{:.6g},{:.6g}\n", ax, comp, exp);
  }
}

} // namespace RMC::callbacks
