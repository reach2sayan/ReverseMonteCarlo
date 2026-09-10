#include <RMC/constraints/CoordinationConstraint.hpp>
#include <RMC/core/Parallel.hpp>
#include <RMC/core/SpeciesIndex.hpp>
#include <algorithm>
#include <limits>
#include <numeric>
#include <ranges>

namespace RMC {

void CoordinationConstraint::add_shell(std::size_t centre,
                                       const std::string &nb_elem, double r_min,
                                       double r_max, int min_cn, int max_cn) {
  shells_.push_back({centre, nb_elem, r_min, r_max, min_cn, max_cn});
  cn_ready_ = false;
  ids_ready_ = false;
}

void CoordinationConstraint::set_elements(
    std::span<const std::string> elements) noexcept {
  elements_ = elements;
  ids_ready_ = false;
  cn_ready_ = false;
}

void CoordinationConstraint::compute_before_move(
    const coords_t &coords,
    std::span<const std::size_t> moved) {
  if (!cn_ready_) {
    full_compute(coords);
  }
  old_cn_ = cn_;
  last_moved_.assign(moved.begin(), moved.end());
  saved_positions_.resize(last_moved_.size());
  std::ranges::transform(
      last_moved_, saved_positions_.begin(), [&](std::size_t k) {
        return coords.row(static_cast<Eigen::Index>(k)).transpose();
      });
  err_before_ = error_from_cn();
}

void CoordinationConstraint::compute_after_move(
    const coords_t &coords,
    std::span<const std::size_t> moved) {
  cn_ = old_cn_;
  incremental_update(coords, moved);
  err_after_ = error_from_cn();
}

void CoordinationConstraint::reject() noexcept {
  ConstraintBase::reject();
  cn_ = old_cn_;
}

double CoordinationConstraint::compute_error(
    const coords_t &coords,
    std::span<const std::size_t> /*moved*/) const noexcept {
  full_compute(coords);
  return error_from_cn();
}

void CoordinationConstraint::build_ids() const {
  const SpeciesIndex sp(elements_);
  elem_id_ = sp.id;
  // A neighbour element absent from the structure gets an id no atom has.
  shell_nb_id_ = shells_ | std::views::transform([&](const Shell &sh) {
                   return sp.id_of(sh.neighbour_elem)
                       .value_or(std::numeric_limits<uint8_t>::max());
                 }) |
                 std::ranges::to<std::vector>();
  ids_ready_ = true;
}

void CoordinationConstraint::full_compute(const coords_t &coords) const
    noexcept {
  if (!ids_ready_ && !elements_.empty()) {
    build_ids();
  }
  const std::size_t N = static_cast<std::size_t>(coords.rows());
  cn_.resize(shells_.size());
  const auto ns = static_cast<std::ptrdiff_t>(shells_.size());
  std::vector<std::ptrdiff_t> shell_idx(static_cast<std::size_t>(ns));
  std::iota(shell_idx.begin(), shell_idx.end(), std::ptrdiff_t{0});
  parallel::for_each(
      shell_idx.begin(), shell_idx.end(), [&](std::ptrdiff_t si) {
        cn_[static_cast<std::size_t>(si)] =
            count_shell(coords, static_cast<std::size_t>(si), N);
      });
  cn_ready_ = true;
}

int CoordinationConstraint::count_shell(const coords_t &coords, std::size_t si,
                                        std::size_t N) const noexcept {
  const auto &sh = shells_[si];
  const double r2_min = sh.r_min * sh.r_min;
  const double r2_max = sh.r_max * sh.r_max;
  const bool filter = !elem_id_.empty();
  const uint8_t nb_id = filter ? shell_nb_id_[si] : uint8_t{0};
  return static_cast<int>(std::ranges::count_if(
      std::views::iota(std::size_t{0}, N), [&](std::size_t j) {
        if (j == sh.centre_idx || absent(j)) {
          return false;
        }
        if (filter && elem_id_[j] != nb_id) {
          return false;
        }
        const double d2 = distance_sq(coords, sh.centre_idx, j);
        return d2 >= r2_min && d2 <= r2_max;
      }));
}

void CoordinationConstraint::incremental_update(
    const coords_t &coords, std::span<const std::size_t> moved) const noexcept {
  const std::size_t N = static_cast<std::size_t>(coords.rows());

  for (const auto [k, old_k] : std::views::zip(moved, saved_positions_)) {
    const vec3_t new_k = coords.row(static_cast<Eigen::Index>(k)).transpose();
    const uint8_t k_id = elem_id_.empty() ? 0 : elem_id_[k];

    for (const auto [si, sh] : shells_ | std::views::enumerate) {
      if (sh.centre_idx == k) {
        cn_[si] = count_shell(coords, static_cast<std::size_t>(si), N);
        continue;
      }

      if (!elem_id_.empty() && k_id != shell_nb_id_[si]) {
        continue;
      }

      const vec3_t j_pos =
          coords.row(static_cast<Eigen::Index>(sh.centre_idx)).transpose();
      const double r2_min = sh.r_min * sh.r_min;
      const double r2_max = sh.r_max * sh.r_max;

      auto in_shell = [&](const vec3_t &pk) {
        vec3_t d = pk - j_pos;
        if (bc_) {
          d = bc_->min_image(d);
        }
        const double d2 = d.squaredNorm();
        return d2 >= r2_min && d2 <= r2_max;
      };

      const bool was_in = in_shell(old_k);
      // A removed atom leaves every shell it was in. Removal does not move the
      // atom, so the geometric test alone (old_k == new_k) would miss it.
      const bool is_in = !absent(k) && in_shell(new_k);
      if (was_in && !is_in) {
        --cn_[si];
      } else if (!was_in && is_in) {
        ++cn_[si];
      }
    }
  }
}

double CoordinationConstraint::error_from_cn() const noexcept {
  return std::ranges::fold_left(
      std::views::zip(cn_, shells_), 0.0, [this](double err, const auto &t) {
        const auto &[cn, sh] = t;
        if (absent(sh.centre_idx)) {
          return err; // a removed centre has no coordination requirement
        }
        if (cn < sh.min_cn) {
          return err + static_cast<double>(sh.min_cn - cn);
        }
        if (cn > sh.max_cn) {
          return err + static_cast<double>(cn - sh.max_cn);
        }
        return err;
      });
}

} // namespace RMC
