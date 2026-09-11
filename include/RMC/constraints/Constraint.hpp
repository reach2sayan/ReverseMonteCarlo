#pragma once
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_erasure/any.hpp>
#include <boost/type_erasure/builtin.hpp>
#include <boost/type_erasure/member.hpp>
#include <cmath>
#include <span>
#include <string_view>

namespace RMC {

namespace detail {
BOOST_TYPE_ERASURE_MEMBER(compute_before_move)
BOOST_TYPE_ERASURE_MEMBER(compute_after_move)
BOOST_TYPE_ERASURE_MEMBER(accept)
BOOST_TYPE_ERASURE_MEMBER(reject)
BOOST_TYPE_ERASURE_MEMBER(standard_error)
BOOST_TYPE_ERASURE_MEMBER(standard_error_before)
BOOST_TYPE_ERASURE_MEMBER(should_reject)
BOOST_TYPE_ERASURE_MEMBER(name)
BOOST_TYPE_ERASURE_MEMBER(computation_cost)
BOOST_TYPE_ERASURE_MEMBER(set_boundary_conditions)
BOOST_TYPE_ERASURE_MEMBER(set_collector)
BOOST_TYPE_ERASURE_MEMBER(is_rigid)
BOOST_TYPE_ERASURE_MEMBER(is_singular)
BOOST_TYPE_ERASURE_MEMBER(set_n_frames)
BOOST_TYPE_ERASURE_MEMBER(set_active_frame)
BOOST_TYPE_ERASURE_MEMBER(initialise)
} // namespace detail

// Satisfied by any type that implements the constraint interface (defaults for
// most of it come from ConstraintBase below).
template <typename T>
concept CConstraint =
    requires(T &c, const coords_t &coords, std::span<const std::size_t> moved,
             const BoundaryConditions &bc, const AtomsCollector *col,
             std::size_t k) {
      c.compute_before_move(coords, moved);
      c.compute_after_move(coords, moved);
      c.accept();
      c.reject();
      { c.standard_error() } -> std::convertible_to<double>;
      { c.standard_error_before() } -> std::convertible_to<double>;
      { c.should_reject() } -> std::convertible_to<bool>;
      { c.name() } -> std::convertible_to<std::string_view>;
      { c.computation_cost() } -> std::convertible_to<double>;
      c.set_boundary_conditions(bc);
      c.set_collector(col);
      { c.is_rigid() } -> std::convertible_to<bool>;
      { c.is_singular() } -> std::convertible_to<bool>;
      c.set_n_frames(k);
      c.set_active_frame(k);
      c.initialise();
    };

// A constraint held by value (Boost.TypeErasure); any CConstraint converts to
// it. standard_error_before() is the soft error BEFORE the move (0 for rigid).
using Constraint = boost::type_erasure::any<boost::mpl::vector<
    boost::type_erasure::copy_constructible<>, boost::type_erasure::typeid_<>,
    boost::type_erasure::relaxed,
    detail::has_compute_before_move<void(const coords_t &,
                                         std::span<const std::size_t>)>,
    detail::has_compute_after_move<void(const coords_t &,
                                        std::span<const std::size_t>)>,
    detail::has_accept<void()>, detail::has_reject<void()>,
    detail::has_standard_error<double(), const boost::type_erasure::_self>,
    detail::has_standard_error_before<double(),
                                      const boost::type_erasure::_self>,
    detail::has_should_reject<bool(), const boost::type_erasure::_self>,
    detail::has_name<std::string_view(), const boost::type_erasure::_self>,
    detail::has_computation_cost<double(), const boost::type_erasure::_self>,
    detail::has_set_boundary_conditions<void(const BoundaryConditions &)>,
    detail::has_set_collector<void(const AtomsCollector *)>,
    detail::has_is_rigid<bool(), const boost::type_erasure::_self>,
    detail::has_is_singular<bool(), const boost::type_erasure::_self>,
    detail::has_set_n_frames<void(std::size_t)>,
    detail::has_set_active_frame<void(std::size_t)>,
    detail::has_initialise<void()>>>;

// Penalty for a value outside [lo, hi]: distance to the nearest endpoint.
[[nodiscard]] FORCE_INLINE constexpr double
range_violation(double x, double lo, double hi) noexcept {
  if (x < lo) {
    return lo - x;
  } else if (x > hi) {
    return x - hi;
  }
  return 0.0;
}

template <typename Derived> class ConstraintBase {
protected:
  const BoundaryConditions *bc_{nullptr};
  const AtomsCollector *collector_{nullptr};
  double err_before_{0.0};
  double err_after_{0.0};

public:
  constexpr void set_boundary_conditions(const BoundaryConditions &bc) noexcept {
    bc_ = &bc;
  }
  // The engine's AtomsCollector (null when off); lets loops skip atoms staged
  // for removal.
  constexpr void set_collector(const AtomsCollector *c) noexcept {
    collector_ = c;
  }

  void compute_before_move(const coords_t &coords,
                           std::span<const std::size_t> moved) {
    err_before_ = static_cast<Derived *>(this)->compute_error(coords, moved);
  }
  void compute_after_move(const coords_t &coords,
                          std::span<const std::size_t> moved) {
    err_after_ = static_cast<Derived *>(this)->compute_error(coords, moved);
  }
  constexpr void accept() noexcept { err_before_ = err_after_; }
  constexpr void reject() noexcept {
    // Reset err_after_ so total stays consistent if compute_after_move was skipped.
    err_after_ = err_before_;
  }

  [[nodiscard]] constexpr double standard_error() const noexcept {
    return err_after_;
  }
  [[nodiscard]] constexpr double standard_error_before() const noexcept {
    return err_before_;
  }
  // Per-constraint downhill test; used only as the RIGID hard gate. The soft
  // accept/reject decision is made by the engine's Sampler on the TOTAL error.
  [[nodiscard]] constexpr bool should_reject() const noexcept {
    return err_after_ > err_before_;
  }
  // Default cost: O(1) or O(bonds). Override in O(N) / O(N²) constraints.
  [[nodiscard]] static constexpr double computation_cost() noexcept {
    return 1.0;
  }
  [[nodiscard]] static constexpr bool is_rigid() noexcept { return false; }
  [[nodiscard]] static constexpr bool is_singular() noexcept { return false; }
  // No-op defaults for multi-frame support; override in pair constraints.
  constexpr void set_n_frames(std::size_t) noexcept {}
  constexpr void set_active_frame(std::size_t) noexcept {}
  // No-op default; pair constraints override to fill their tables.
  constexpr void initialise() noexcept {}

protected:
  // True when atom `i` is staged/committed for removal (terms involving it skip).
  // Null collector short-circuits before any lookup.
  [[nodiscard]] FORCE_INLINE bool absent(std::size_t i) const noexcept {
    return collector_ != nullptr && collector_->absent(i);
  }
  [[nodiscard]] FORCE_INLINE double
  distance_sq(const coords_t &c, std::size_t i, std::size_t j) const noexcept {
    const vec3_t d = c.row(j).transpose() - c.row(i).transpose();
    return (bc_ ? bc_->min_image(d) : d).squaredNorm();
  }
  [[nodiscard]] FORCE_INLINE double distance(const coords_t &c, std::size_t i,
                                             std::size_t j) const noexcept {
    return std::sqrt(distance_sq(c, i, j));
  }
};

// Hard gate: rejects moves that worsen it, but standard_error() == 0 so it
// does NOT contribute to the engine's total chi²
template <typename Derived>
class RigidConstraintBase : public ConstraintBase<Derived> {
public:
  [[nodiscard]] static constexpr double standard_error() noexcept { return 0.0; }
  [[nodiscard]] static constexpr double standard_error_before() noexcept {
    return 0.0;
  }
  [[nodiscard]] static constexpr bool is_rigid() noexcept { return true; }
};

// Only one instance of this constraint type is allowed per ConstraintCollection
// standard_error() is NOT overridden —
// singular constraints still contribute to total chi².
template <typename Derived>
class SingularConstraintBase : public ConstraintBase<Derived> {
public:
  [[nodiscard]] static constexpr bool is_singular() noexcept { return true; }
};

} // namespace RMC
