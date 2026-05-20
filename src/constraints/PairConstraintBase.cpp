#include <RMC/constraints/PairHistogram.hpp>
#include <cmath>
#include <numbers>
#include <unordered_map>

namespace RMC {

void accumulate_pair_histogram(
    vec_t &hist, const coords_t &coords, const BoundaryConditions *bc,
    const std::vector<uint8_t> &elem_id, const PairWeightTable &weight_table,
    double r_min, double r_max, double bin_width, int n_bins,
    std::span<const std::size_t> molecule_ids, bool exclude_intra) {
  const Eigen::Index N = coords.rows();
  const bool weighted = !weight_table.empty();
  const bool filter_intra = exclude_intra && !molecule_ids.empty();

#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
  std::vector<vec_t> partial(static_cast<std::size_t>(nthreads),
                             vec_t::Zero(n_bins));

#pragma omp parallel for schedule(static)
  for (Eigen::Index i = 0; i < N - 1; ++i) {
    vec_t &local = partial[static_cast<std::size_t>(omp_get_thread_num())];
    for (Eigen::Index j = i + 1; j < N; ++j) {
      if (filter_intra && molecule_ids[static_cast<std::size_t>(i)] ==
                              molecule_ids[static_cast<std::size_t>(j)]) {
        continue;
      }
      vec3_t delta = coords.row(j).transpose() - coords.row(i).transpose();
      if (bc) {
        delta = bc_min_image(*bc, delta);
      }
      const double d = delta.norm();
      if (d < r_min || d >= r_max) {
        continue;
      }
      const int bin = static_cast<int>((d - r_min) / bin_width);
      if (bin < 0 || bin >= n_bins) {
        continue;
      }
      double w = 1.0;
      if (weighted) {
        PairIdKey key{elem_id[static_cast<std::size_t>(i)],
                      elem_id[static_cast<std::size_t>(j)]};
        if (auto it = weight_table.find(key); it != weight_table.end()) {
          w = it->second;
        }
      }
      local(bin) += 2.0 * w;
    }
  }

  for (auto &p : partial) {
    hist += p;
  }

#else
  for (auto [i, j] : upper_triangle_pairs(N)) {
    if (filter_intra && molecule_ids[static_cast<std::size_t>(i)] ==
                            molecule_ids[static_cast<std::size_t>(j)]) {
      continue;
    }
    vec3_t delta = coords.row(j).transpose() - coords.row(i).transpose();
    if (bc) {
      delta = bc_min_image(*bc, delta);
    }
    const double d = delta.norm();
    if (d < r_min || d >= r_max) {
      continue;
    }
    const int bin = static_cast<int>((d - r_min) / bin_width);
    if (bin < 0 || bin >= n_bins) {
      continue;
    }
    double w = 1.0;
    if (weighted) {
      PairIdKey key{elem_id[static_cast<std::size_t>(i)],
                    elem_id[static_cast<std::size_t>(j)]};
      if (auto it = weight_table.find(key); it != weight_table.end()) {
        w = it->second;
      }
    }
    hist(bin) += 2.0 * w;
  }
#endif
}

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
    if (ins) {
      ++next_id;
    }
    elem_id_[i] = it->second;
  }
  weight_table_.clear();
  for (const auto &[key, w] : weights_) {
    auto ia = name_to_id.find(key.a);
    auto ib = name_to_id.find(key.b);
    if (ia != name_to_id.end() && ib != name_to_id.end()) {
      weight_table_[PairIdKey{ia->second, ib->second}] = w;
    }
  }
}

} // namespace RMC
