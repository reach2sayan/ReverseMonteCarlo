#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <cmath>
#include <unordered_map>
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
//
// Element filtering uses integer IDs built at first use — avoids std::string
// comparisons in the inner loop.
class CoordinationConstraint : public RigidConstraintBase<CoordinationConstraint> {
public:
  struct Shell {
    std::size_t centre_idx;
    std::string neighbour_elem;
    double r_min{0.0};
    double r_max{3.0};
    int min_cn{0};
    int max_cn{12};
  };

  void add_shell(std::size_t centre, const std::string &nb_elem, double r_min,
                 double r_max, int min_cn, int max_cn) {
    shells_.push_back({centre, nb_elem, r_min, r_max, min_cn, max_cn});
    cn_ready_ = false;
    ids_ready_ = false;
  }

  void set_elements(std::span<const std::string> elements) noexcept {
    elements_ = elements;
    ids_ready_ = false;
    cn_ready_ = false;
  }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "CoordinationConstraint";
  }
  [[nodiscard]] static constexpr double
  computation_cost(IConstraint::Token) noexcept {
    return 10.0;
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

  void accept(IConstraint::Token tok) noexcept { ConstraintBase::accept(tok); }

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

  // Integer element IDs — built lazily, avoids string comparisons in hot paths.
  mutable std::vector<uint8_t> elem_id_; // elem_id_[i] = ID of atom i's element
  mutable std::vector<uint8_t>
      shell_nb_id_; // shell_nb_id_[si] = ID of shell si's neighbour_elem
  mutable bool ids_ready_{false};

  mutable std::vector<int> cn_;
  mutable std::vector<int> old_cn_;
  mutable bool cn_ready_{false};
  mutable std::vector<vec3_t> saved_positions_;
  mutable std::vector<std::size_t> last_moved_;

  void build_ids() const {
    std::unordered_map<std::string, uint8_t> name_to_id;
    uint8_t next_id = 0;
    elem_id_.resize(elements_.size());
    for (std::size_t i = 0; i < elements_.size(); ++i) {
      auto [it, ins] = name_to_id.try_emplace(elements_[i], next_id);
      if (ins)
        ++next_id;
      elem_id_[i] = it->second;
    }
    shell_nb_id_.resize(shells_.size());
    for (std::size_t si = 0; si < shells_.size(); ++si) {
      auto it = name_to_id.find(shells_[si].neighbour_elem);
      shell_nb_id_[si] = (it != name_to_id.end())
                             ? it->second
                             : std::numeric_limits<uint8_t>::max();
    }
    ids_ready_ = true;
  }

  void full_compute(const coords_t &coords) const noexcept {
    if (!ids_ready_ && !elements_.empty())
      build_ids();
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
    if (elem_id_.empty()) {
      // No element filtering.
      for (std::size_t j = 0; j < N; ++j) {
        if (j == sh.centre_idx)
          continue;
        const double d2 = distance_sq(coords, sh.centre_idx, j);
        if (d2 >= r2_min && d2 <= r2_max)
          ++cn;
      }
    } else {
      const uint8_t nb_id = shell_nb_id_[si];
      for (std::size_t j = 0; j < N; ++j) {
        if (j == sh.centre_idx)
          continue;
        if (elem_id_[j] != nb_id)
          continue;
        const double d2 = distance_sq(coords, sh.centre_idx, j);
        if (d2 >= r2_min && d2 <= r2_max)
          ++cn;
      }
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
      const uint8_t k_id = elem_id_.empty() ? 0 : elem_id_[k];

      for (std::size_t si = 0; si < ns; ++si) {
        const auto &sh = shells_[si];

        if (sh.centre_idx == k) {
          cn_[si] = count_shell(coords, si, N);
          continue;
        }

        if (!elem_id_.empty() && k_id != shell_nb_id_[si])
          continue;

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
