#include <RMC/constraints/ConstraintCollection.hpp>
#include <boost/assert.hpp>

namespace RMC {

void ConstraintCollection::add(Constraint c) {
  BOOST_ASSERT_MSG(!c.is_singular() ||
                       std::ranges::none_of(constraints_,
                                            [&c](const Constraint &e) {
                                              return e.is_singular() &&
                                                     e.name() == c.name();
                                            }),
                   "ConstraintCollection: duplicate singular constraint");
  if (collector_ != nullptr) {
    c.set_collector(collector_);
  }
  const auto at = std::ranges::lower_bound(
      constraints_, c.computation_cost(), {},
      [](const Constraint &x) { return x.computation_cost(); });
  constraints_.insert(at, std::move(c));
}

void ConstraintCollection::set_collector(const AtomsCollector *c) noexcept {
  collector_ = c;
  each([c](Constraint &x) { x.set_collector(c); });
}

void ConstraintCollection::compute_before_move(
    const coords_t &coords, std::span<const std::size_t> moved) {
  each([&](Constraint &c) { c.compute_before_move(coords, moved); });
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

bool ConstraintCollection::rigid_should_reject() const noexcept {
  return std::ranges::any_of(constraints_, [](const Constraint &c) {
    return c.is_rigid() && c.should_reject();
  });
}

void ConstraintCollection::accept() noexcept {
  each([](Constraint &c) { c.accept(); });
}

void ConstraintCollection::reject() noexcept {
  each([](Constraint &c) { c.reject(); });
}

double ConstraintCollection::total_error() const noexcept {
  return sum([](const Constraint &c) { return c.standard_error(); });
}

double ConstraintCollection::total_error_before() const noexcept {
  return sum([](const Constraint &c) { return c.standard_error_before(); });
}

std::vector<std::pair<std::string_view, double>>
ConstraintCollection::error_breakdown() const {
  return constraints_ | std::views::transform([](const Constraint &c) {
           return std::pair{c.name(), c.standard_error()};
         }) |
         std::ranges::to<std::vector>();
}

void ConstraintCollection::initialise_all() {
  each([](Constraint &c) { c.initialise(); });
}

void ConstraintCollection::set_n_frames(std::size_t n) noexcept {
  each([n](Constraint &c) { c.set_n_frames(n); });
}

void ConstraintCollection::set_active_frame(std::size_t k) noexcept {
  each([k](Constraint &c) { c.set_active_frame(k); });
}

} // namespace RMC
