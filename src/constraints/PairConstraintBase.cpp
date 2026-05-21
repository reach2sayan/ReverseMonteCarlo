#include <RMC/constraints/PairHistogram.hpp>
#include <boost/histogram.hpp>
#include <cmath>
#include <numbers>
#include <unordered_map>

#if defined(RMC_USE_TBB)
#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>
#endif

namespace RMC {

void accumulate_pair_histogram(vec_t &hist, const coords_t &coords,
                               const BoundaryConditions *bc,
                               const std::vector<uint8_t> &elem_id,
                               const PairWeightTable &weight_table,
                               double r_min, double r_max, int n_bins,
                               std::span<const std::size_t> molecule_ids,
                               bool exclude_intra) {
  namespace bh = boost::histogram;
  auto make_histogram = [&] {
    return bh::make_histogram_with(bh::dense_storage<double>{},
                                   bh::axis::regular<>(n_bins, r_min, r_max));
  };

  const Eigen::Index N = coords.rows();
  const bool weighted = !weight_table.empty();
  const bool filter_intra = exclude_intra && !molecule_ids.empty();

#if defined(RMC_USE_TBB)
  // TBB thread pool — no per-call spawn cost. Each row i gets its own
  // private histogram; they are reduced at the end.
  std::vector<Eigen::Index> rows(static_cast<std::size_t>(N - 1));
  std::iota(rows.begin(), rows.end(), Eigen::Index{0});
  std::vector<decltype(make_histogram())> partial(
      static_cast<std::size_t>(N - 1), make_histogram());
  std::for_each(
      std::execution::par_unseq, rows.begin(), rows.end(), [&](Eigen::Index i) {
        auto &local = partial[static_cast<std::size_t>(i)];
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
          double w = 1.0;
          if (weighted) {
            PairIdKey key{elem_id[static_cast<std::size_t>(i)],
                          elem_id[static_cast<std::size_t>(j)]};
            if (auto it = weight_table.find(key); it != weight_table.end()) {
              w = it->second;
            }
          }
          local(bh::weight(2.0 * w), d);
        }
      });
  auto h = make_histogram();
  for (auto &p : partial) {
    h += p;
  }

#elif defined(_OPENMP)
  const int nthreads = omp_get_max_threads();
  std::vector<decltype(make_histogram())> partial(
      static_cast<std::size_t>(nthreads), make_histogram());

#pragma omp parallel for schedule(static)
  for (Eigen::Index i = 0; i < N - 1; ++i) {
    auto &local = partial[static_cast<std::size_t>(omp_get_thread_num())];
    for (Eigen::Index j = i + 1; j < N; ++j) {
      if (filter_intra && molecule_ids[static_cast<std::size_t>(i)] ==
                              molecule_ids[static_cast<std::size_t>(j)])
        continue;
      vec3_t delta = coords.row(j).transpose() - coords.row(i).transpose();
      if (bc) {
        delta = bc_min_image(*bc, delta);
      }
      const double d = delta.norm();
      double w = 1.0;
      if (weighted) {
        PairIdKey key{elem_id[static_cast<std::size_t>(i)],
                      elem_id[static_cast<std::size_t>(j)]};
        if (auto it = weight_table.find(key); it != weight_table.end()) {
          w = it->second;
        }
      }
      local(bh::weight(2.0 * w), d);
    }
  }
  auto h = make_histogram();
  for (auto &p : partial) {
    h += p;
  }
#else
  auto h = make_histogram();
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
    double w = 1.0;
    if (weighted) {
      PairIdKey key{elem_id[static_cast<std::size_t>(i)],
                    elem_id[static_cast<std::size_t>(j)]};
      if (auto it = weight_table.find(key); it != weight_table.end()) {
        w = it->second;
      }
    }
    h(bh::weight(2.0 * w), d);
  }
#endif
  for (int k = 0; k < n_bins; ++k) {
    hist(k) = h[k];
  }
}

void accumulate_moved_pairs(
    vec_t &hist, const coords_t &coords, const BoundaryConditions *bc,
    const std::vector<uint8_t> &elem_id, const PairWeightTable &weight_table,
    double r_min, double r_max, int n_bins, std::span<const std::size_t> moved,
    std::span<const std::size_t> molecule_ids, bool exclude_intra) {
  namespace bh = boost::histogram;

  auto h = bh::make_histogram_with(bh::dense_storage<double>{},
                                   bh::axis::regular<>(n_bins, r_min, r_max));

  const Eigen::Index N = coords.rows();
  const bool weighted = !weight_table.empty();
  const bool filter_intra = exclude_intra && !molecule_ids.empty();

  for (std::size_t mk = 0; mk < moved.size(); ++mk) {
    const std::size_t k = moved[mk];
    for (Eigen::Index jj = 0; jj < N; ++jj) {
      const std::size_t j = static_cast<std::size_t>(jj);
      if (j == k) {
        continue;
      }
      // Avoid double-counting pairs where both atoms are in `moved`:
      // count (k,j) only when k appears before j in the moved array.
      bool already_counted = false;
      for (std::size_t prev = 0; prev < mk; ++prev) {
        if (moved[prev] == j) {
          already_counted = true;
          break;
        }
      }
      if (already_counted ||
          (filter_intra && molecule_ids[k] == molecule_ids[j])) {
        continue;
      }

      vec3_t delta = coords.row(jj).transpose() -
                     coords.row(static_cast<Eigen::Index>(k)).transpose();
      if (bc) {
        delta = bc_min_image(*bc, delta);
      }
      const double d = delta.norm();

      double w = 1.0;
      if (weighted) {
        PairIdKey key{elem_id[k], elem_id[j]};
        if (auto it = weight_table.find(key); it != weight_table.end())
          w = it->second;
      }
      h(bh::weight(2.0 * w), d);
    }
  }

  for (int k = 0; k < n_bins; ++k) {
    hist(k) += h[k];
  }
}

void PairConstraintBase::set_experimental_data(const mat_t &data) {
  BOOST_ASSERT_MSG(data.cols() >= 2, "PairConstraint: need 2-column r/data");
  const Eigen::Index N = data.rows();
  exp_r_ = data.col(0);
  exp_data_ = data.col(1);
  r_min_ = exp_r_(0);
  r_max_ = exp_r_(N - 1);
  bin_width_ = (N > 1) ? (exp_r_(1) - exp_r_(0)) : 0.1;
  n_bins_ = static_cast<int>(N);
  computed_.resize(N);
}

void PairConstraintBase::initialise() {
  const auto idx = Eigen::ArrayXd::LinSpaced(n_bins_, 0, n_bins_ - 1);
  const auto r_lo = r_min_ + idx * bin_width_;
  const auto r_hi = r_lo + bin_width_;
  shell_vols_ = (4.0 * std::numbers::pi / 3.0) * (r_hi.cube() - r_lo.cube());

  std::unordered_map<std::string_view, uint8_t> name_to_id;
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
