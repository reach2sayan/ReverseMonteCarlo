#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <cmath>
#include <vector>

namespace RMC {

// Per-atom coordination constraint: atom i must have between min_cn and max_cn
// neighbours of element `neighbour_elem` within [r_min, r_max].
// std_err = Σ max(0, cn_i - max_cn) + max(0, min_cn - cn_i)
//
// Incremental: cn_ is cached between steps. On a move of atom k:
//   - Shell centred at k: full recompute (O(N))
//   - All other shells: ±1 update based on k entering/leaving (O(N_shells))
// Total cost per step: O(N) instead of O(N²).
class CoordinationConstraint : public ConstraintBase<CoordinationConstraint> {
public:
  struct Shell {
    std::size_t centre_idx;
    std::string neighbour_elem;
    double r_min{0.0};
    double r_max{3.0};
    int min_cn{0};
    int max_cn{12};
  };

  constexpr void add_shell(std::size_t centre, const std::string &nb_elem,
                           double r_min, double r_max, int min_cn, int max_cn) {
    shells_.push_back({centre, nb_elem, r_min, r_max, min_cn, max_cn});
    cn_ready_ = false;
  }

  constexpr void set_elements(std::span<const std::string> elements) noexcept {
    elements_ = elements;
  }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "CoordinationConstraint";
  }

  // Override ConstraintBase defaults so each step is O(N) not O(N²).
  void compute_before_move(IConstraint::Token, const coords_t &coords,
                           std::span<const std::size_t> moved) {
    if (!cn_ready_)
      full_compute(coords);
    old_cn_ = cn_;
    last_moved_.assign(moved.begin(), moved.end());
    saved_positions_.clear();
    for (auto k : last_moved_) {
      saved_positions_.push_back(
          coords.row(static_cast<Eigen::Index>(k)).transpose());
    }
    err_before_ = error_from_cn();
  }

  void compute_after_move(IConstraint::Token, const coords_t &coords,
                          std::span<const std::size_t> moved) {
    cn_ = old_cn_;
    incremental_update(coords, moved);
    err_after_ = error_from_cn();
  }

  constexpr void accept(IConstraint::Token tok) noexcept {
    ConstraintBase::accept(tok);
  }

  void reject(IConstraint::Token tok) noexcept {
    ConstraintBase::reject(tok);
    cn_ = old_cn_;
  }

  // Fallback full-recompute; used by tests that call compute_error directly.
  [[nodiscard]] double
  compute_error(const coords_t &coords,
                std::span<const std::size_t> /*moved*/) const noexcept {
    full_compute(coords);
    return error_from_cn();
  }

private:
  std::vector<Shell> shells_;
  std::span<const std::string> elements_;

  mutable std::vector<int> cn_;
  mutable std::vector<int> old_cn_;
  mutable bool cn_ready_{false};
  mutable std::vector<vec3_t> saved_positions_;
  mutable std::vector<std::size_t> last_moved_;

  void full_compute(const coords_t &coords) const noexcept {
    const std::size_t N = static_cast<std::size_t>(coords.rows());
    cn_.resize(shells_.size());
    for (std::size_t si = 0; si < shells_.size(); ++si)
      cn_[si] = count_shell(coords, si, N);
    cn_ready_ = true;
  }

  int count_shell(const coords_t &coords, std::size_t si,
                  std::size_t N) const noexcept {
    const auto &sh = shells_[si];
    const double r2_min = sh.r_min * sh.r_min;
    const double r2_max = sh.r_max * sh.r_max;
    int cn = 0;
    for (std::size_t j = 0; j < N; ++j) {
      if (j == sh.centre_idx)
        continue;
      if (!elements_.empty() && elements_[j] != sh.neighbour_elem)
        continue;
      const double d2 = distance_sq(coords, sh.centre_idx, j);
      if (d2 >= r2_min && d2 <= r2_max)
        ++cn;
    }
    return cn;
  }

  void incremental_update(const coords_t &coords,
                          std::span<const std::size_t> moved) const noexcept {
    const std::size_t N = static_cast<std::size_t>(coords.rows());
    const std::size_t ns = shells_.size();

    for (std::size_t ki = 0; ki < moved.size(); ++ki) {
      const std::size_t k = moved[ki];
      const vec3_t &old_k = saved_positions_[ki];
      const vec3_t new_k = coords.row(static_cast<Eigen::Index>(k)).transpose();

      for (std::size_t si = 0; si < ns; ++si) {
        const auto &sh = shells_[si];

        if (sh.centre_idx == k) {
          // All distances from k changed — full recompute of this shell.
          cn_[si] = count_shell(coords, si, N);
          continue;
        }

        // Shell centred at j ≠ k: only k's membership may have changed.
        if (!elements_.empty() && elements_[k] != sh.neighbour_elem)
          continue;

        // j did not move so coords.row(j) is correct for both before and after.
        const vec3_t j_pos =
            coords.row(static_cast<Eigen::Index>(sh.centre_idx)).transpose();
        const double r2_min = sh.r_min * sh.r_min;
        const double r2_max = sh.r_max * sh.r_max;

        auto in_shell = [&](const vec3_t &pk) {
          vec3_t d = pk - j_pos;
          if (bc_)
            d = bc_min_image(*bc_, d);
          const double d2 = d.squaredNorm();
          return d2 >= r2_min && d2 <= r2_max;
        };

        const bool was_in = in_shell(old_k);
        const bool is_in = in_shell(new_k);
        if (was_in && !is_in)
          --cn_[si];
        else if (!was_in && is_in)
          ++cn_[si];
      }
    }
  }

  double error_from_cn() const noexcept {
    double err = 0.0;
    for (std::size_t si = 0; si < shells_.size(); ++si) {
      const auto &sh = shells_[si];
      if (cn_[si] < sh.min_cn)
        err += static_cast<double>(sh.min_cn - cn_[si]);
      else if (cn_[si] > sh.max_cn)
        err += static_cast<double>(cn_[si] - sh.max_cn);
    }
    return err;
  }
};

} // namespace RMC
