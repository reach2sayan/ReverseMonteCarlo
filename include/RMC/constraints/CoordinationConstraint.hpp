#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <cstdint>
#include <string>
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
class CoordinationConstraint
    : public RigidConstraintBase<CoordinationConstraint> {
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
                 double r_max, int min_cn, int max_cn);

  void set_elements(std::span<const std::string> elements) noexcept;

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "CoordinationConstraint";
  }
  [[nodiscard]] static constexpr double
  computation_cost() noexcept {
    return 10.0;
  }

  // Override ConstraintBase defaults so each step is O(N) not O(N²).
  void compute_before_move(const coords_t &coords,
                           std::span<const std::size_t> moved);

  void compute_after_move(const coords_t &coords,
                          std::span<const std::size_t> moved);

  void accept() noexcept { ConstraintBase::accept(); }

  void reject() noexcept;

  // Fallback full-recompute; used by tests that call compute_error directly.
  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const
      noexcept;

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

  void build_ids() const;
  void full_compute(const coords_t &coords) const noexcept;
  int count_shell(const coords_t &coords, std::size_t si,
                  std::size_t N) const noexcept;
  void incremental_update(const coords_t &coords,
                          std::span<const std::size_t> moved) const noexcept;
  double error_from_cn() const noexcept;
};

} // namespace RMC
