#pragma once
#include <fullrmc/constraints/Constraint.hpp>
#include <vector>

namespace fullrmc {

class ConstraintCollection {
public:
  void add(IConstraint c) {
    if (bc_)
      c.set_boundary_conditions(*bc_);
    constraints_.push_back(std::move(c));
  }

  void set_boundary_conditions(const BoundaryConditions &bc) noexcept {
    bc_ = &bc;
    for (auto &c : constraints_)
      c.set_boundary_conditions(bc);
  }

  void compute_before_move(const coords_t &coords,
                           std::span<const std::size_t> moved) {
    for (auto &c : constraints_)
      c.compute_before_move(coords, moved);
  }
  void compute_after_move(const coords_t &coords,
                          std::span<const std::size_t> moved) {
    for (auto &c : constraints_)
      c.compute_after_move(coords, moved);
  }

  [[nodiscard]] bool should_reject() const noexcept {
    for (auto &c : constraints_)
      if (c.should_reject())
        return true;
    return false;
  }

  void accept() noexcept {
    for (auto &c : constraints_)
      c.accept();
  }
  void reject() noexcept {
    for (auto &c : constraints_)
      c.reject();
  }

  [[nodiscard]] double total_error() const noexcept {
    double total = 0.0;
    for (auto &c : constraints_)
      total += c.standard_error();
    return total;
  }

  [[nodiscard]] std::size_t size() const noexcept { return constraints_.size(); }

  [[nodiscard]] IConstraint &operator[](std::size_t i) {
    return constraints_[i];
  }
  [[nodiscard]] const IConstraint &operator[](std::size_t i) const {
    return constraints_[i];
  }

private:
  std::vector<IConstraint> constraints_;
  const BoundaryConditions *bc_{nullptr};
};

} // namespace fullrmc
