#include <RMC/constraints/PairHistogram.hpp>
#include <cmath>
#include <numbers>
#include <unordered_map>

namespace RMC {

void PairConstraintBase::initialise() {
  const auto idx = Eigen::ArrayXd::LinSpaced(n_bins_, 0, n_bins_ - 1);
  const auto r_lo = r_min_ + idx * bin_width_;
  const auto r_hi = r_lo + bin_width_;
  shell_vols_ = (4.0 * std::numbers::pi / 3.0) * (r_hi.cube() - r_lo.cube());

  std::unordered_map<std::string, uint8_t> name_to_id;
  uint8_t next_id = 0;
  elem_id_.resize(elements_.size());
  for (std::size_t i = 0; i < elements_.size(); ++i) {
    auto [it, ins] = name_to_id.try_emplace(elements_[i], next_id);
    if (ins)
      ++next_id;
    elem_id_[i] = it->second;
  }

  weight_table_.clear();
  for (const auto &[key, w] : weights_) {
    auto ia = name_to_id.find(key.a);
    auto ib = name_to_id.find(key.b);
    if (ia != name_to_id.end() && ib != name_to_id.end())
      weight_table_[PairIdKey{ia->second, ib->second}] = w;
  }
}

} // namespace RMC
