#pragma once
#include <cmath>
#include <fullrmc/constraints/Constraint.hpp>
#include <vector>

namespace fullrmc {

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

  void add_shell(std::size_t centre, const std::string &nb_elem, double r_min,
                 double r_max, int min_cn, int max_cn) {
    shells_.push_back({centre, nb_elem, r_min, r_max, min_cn, max_cn});
  }

  void set_elements(const std::vector<std::string> *elements) noexcept {
    elements_ = elements;
  }

  [[nodiscard]] std::string name() const override {
    return "CoordinationConstraint";
  }

  [[nodiscard]] double
  compute_error(const coords_t &coords,
                std::span<const std::size_t> /*moved*/) const {
    double err = 0.0;
    const std::size_t N = static_cast<std::size_t>(coords.rows());
    for (auto &sh : shells_) {
      int cn = 0;
      for (std::size_t j = 0; j < N; ++j) {
        if (j == sh.centre_idx)
          continue;
        if (elements_ && (*elements_)[j] != sh.neighbour_elem)
          continue;
        double d = distance(coords, static_cast<std::size_t>(sh.centre_idx),
                            static_cast<std::size_t>(j));
        if (d >= sh.r_min && d <= sh.r_max)
          ++cn;
      }
      if (cn < sh.min_cn)
        err += static_cast<double>(sh.min_cn - cn);
      else if (cn > sh.max_cn)
        err += static_cast<double>(cn - sh.max_cn);
    }
    return err;
  }

private:
  std::vector<Shell> shells_;
  const std::vector<std::string> *elements_ = nullptr;
};

} // namespace fullrmc
