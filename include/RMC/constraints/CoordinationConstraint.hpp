#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <cmath>
#include <vector>

namespace RMC {

// Per-atom coordination constraint: atom i must have between min_cn and max_cn
// neighbours of element `neighbour_element` within [r_min, r_max].
// std_err = Σ max(0, cn_i - max_cn) + max(0, min_cn - cn_i)
class CoordinationConstraint : public ConstraintBase<CoordinationConstraint> {
public:
  struct Shell {
    std::size_t centre_idx;     // atom index
    std::string neighbour_elem; // element symbol of neighbours
    double r_min{0.0};
    double r_max{3.0};
    int min_cn{0};
    int max_cn{12};
  };

  constexpr void add_shell(std::size_t centre, const std::string &nb_elem,
                           double r_min, double r_max, int min_cn, int max_cn) {
    shells_.push_back({centre, nb_elem, r_min, r_max, min_cn, max_cn});
  }

  constexpr void set_elements(std::span<const std::string> elements) noexcept {
    elements_ = elements;
  }

  [[nodiscard]] std::string name() const { return "CoordinationConstraint"; }

  [[nodiscard]] double
  compute_error(const coords_t &coords,
                std::span<const std::size_t> /*moved*/) const noexcept {
    double err = 0.0;
    const std::size_t N = static_cast<std::size_t>(coords.rows());
    const auto ns = static_cast<std::ptrdiff_t>(shells_.size());
#pragma omp parallel for reduction(+ : err) schedule(static)
    for (std::ptrdiff_t si = 0; si < ns; ++si) {
      const auto &sh = shells_[static_cast<std::size_t>(si)];
      int cn = 0;
      for (std::size_t j = 0; j < N; ++j) {
        if (j == sh.centre_idx) {
          continue;
        }
        if (!elements_.empty() && elements_[j] != sh.neighbour_elem) {
          continue;
        }
        const double d = distance(coords, sh.centre_idx, j);
        if (d >= sh.r_min && d <= sh.r_max) {
          ++cn;
        }
      }
      if (cn < sh.min_cn) {
        err += static_cast<double>(sh.min_cn - cn);
      } else if (cn > sh.max_cn) {
        err += static_cast<double>(cn - sh.max_cn);
      }
    }
    return err;
  }

private:
  std::vector<Shell> shells_;
  std::span<const std::string> elements_;
};

} // namespace RMC
