#include <RMC/constraints/ConstraintCollection.hpp>
#include <algorithm>
#include <boost/assert.hpp>

namespace RMC {

void ConstraintCollection::add(Constraint c) {
  if (c.is_singular()) {
    BOOST_ASSERT_MSG(std::ranges::none_of(constraints_,
                                          [&c](const Constraint &e) {
                                            return e.is_singular() &&
                                                   e.name() == c.name();
                                          }),
                     "ConstraintCollection: duplicate singular constraint");
  }
  if (bc_.has_value()) {
    c.set_boundary_conditions(bc_.value());
  }
  if (collector_ != nullptr) {
    c.set_collector(collector_);
  }
  const double cost = c.computation_cost();
  auto it = std::lower_bound(
      constraints_.begin(), constraints_.end(), cost,
      [](const Constraint &x, double v) { return x.computation_cost() < v; });
  constraints_.insert(it, std::move(c));
}

void ConstraintCollection::set_boundary_conditions(
    const BoundaryConditions &bc) noexcept {
  bc_ = bc;
  std::ranges::for_each(constraints_,
                        [&](Constraint &c) { c.set_boundary_conditions(bc); });
}

void ConstraintCollection::set_collector(const AtomsCollector *c) noexcept {
  collector_ = c;
  std::ranges::for_each(constraints_,
                        [&](Constraint &cc) { cc.set_collector(c); });
}

void ConstraintCollection::compute_before_move(
    const coords_t &coords, std::span<const std::size_t> moved) {
  std::ranges::for_each(constraints_, [&](Constraint &c) {
    c.compute_before_move(coords, moved);
  });
}

void ConstraintCollection::compute_after_move(
    const coords_t &coords, std::span<const std::size_t> moved) {
  for (auto &c : constraints_) {
    c.compute_after_move(coords, moved);
    if (c.is_rigid() && c.should_reject()) {
      break;
    }
  }
}

bool ConstraintCollection::should_reject() const noexcept {
  return std::ranges::any_of(
      constraints_, [](const Constraint &c) { return c.should_reject(); });
}

bool ConstraintCollection::rigid_should_reject() const noexcept {
  return std::ranges::any_of(constraints_, [](const Constraint &c) {
    return c.is_rigid() && c.should_reject();
  });
}

void ConstraintCollection::accept() noexcept {
  std::ranges::for_each(constraints_, [](Constraint &c) { c.accept(); });
}

void ConstraintCollection::reject() noexcept {
  std::ranges::for_each(constraints_, [](Constraint &c) { c.reject(); });
}

double ConstraintCollection::total_error() const noexcept {
  return std::ranges::fold_left(
      constraints_, 0.0,
      [](double acc, const auto &c) { return acc + c.standard_error(); });
}

double ConstraintCollection::total_error_before() const noexcept {
  return std::ranges::fold_left(constraints_, 0.0,
                                [](double acc, const auto &c) {
                                  return acc + c.standard_error_before();
                                });
}

std::vector<std::pair<std::string_view, double>>
ConstraintCollection::error_breakdown() const {
  std::vector<std::pair<std::string_view, double>> out;
  out.reserve(constraints_.size());
  for (const auto &c : constraints_) {
    out.emplace_back(c.name(), c.standard_error());
  }
  return out;
}

void ConstraintCollection::initialise_all() {
  std::ranges::for_each(constraints_, [](Constraint &c) { c.initialise(); });
}

void ConstraintCollection::set_n_frames(std::size_t n) noexcept {
  std::ranges::for_each(constraints_,
                        [n](Constraint &c) { c.set_n_frames(n); });
}

void ConstraintCollection::set_active_frame(std::size_t k) noexcept {
  std::ranges::for_each(constraints_,
                        [k](Constraint &c) { c.set_active_frame(k); });
}

} // namespace RMC
