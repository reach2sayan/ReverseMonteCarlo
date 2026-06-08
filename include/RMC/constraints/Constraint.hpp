#pragma once
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/TypeErasure.hpp>
#include <RMC/core/Types.hpp>
#include <cmath>
#include <memory>
#include <string_view>

namespace RMC {

class Constraint; // forward for friend declaration

namespace detail {
struct ConstraintToken {
private:
  constexpr ConstraintToken() = default;
  friend class ::RMC::Constraint;
};
} // namespace detail

// Satisfied by any type that implements the passkey-gated constraint interface.
template <typename T>
concept CConstraint =
    requires(T &c, detail::ConstraintToken tok, const coords_t &coords,
             std::span<const std::size_t> moved, const BoundaryConditions &bc,
             const AtomsCollector *col, std::size_t k) {
      c.compute_before_move(tok, coords, moved);
      c.compute_after_move(tok, coords, moved);
      c.accept(tok);
      c.reject(tok);
      { c.standard_error(tok) } -> std::convertible_to<double>;
      { c.standard_error_before(tok) } -> std::convertible_to<double>;
      { c.should_reject(tok) } -> std::convertible_to<bool>;
      { c.name() } -> std::convertible_to<std::string_view>;
      { c.computation_cost(tok) } -> std::convertible_to<double>;
      c.set_boundary_conditions(tok, bc);
      c.set_collector(tok, col);
      { c.is_rigid(tok) } -> std::convertible_to<bool>;
      { c.is_singular(tok) } -> std::convertible_to<bool>;
      c.set_n_frames(tok, k);
      c.set_active_frame(tok, k);
      c.initialise(tok);
    };

// standard_error_before() is the soft error BEFORE the proposed move (0 for
// rigid constraints) — lets the engine form the total ΔE handed to the Sampler.
// name() is the only method forwarded without the passkey token.
#define RMC_CONSTRAINT_METHODS                                                 \
  ((0, void, compute_before_move,                                              \
    (const coords_t &coords, std::span<const std::size_t> moved), 2,           \
    (coords, moved), , , WITH_TOKEN))(                                         \
      (0, void, compute_after_move,                                            \
       (const coords_t &coords, std::span<const std::size_t> moved), 2,        \
       (coords, moved), , ,                                                    \
       WITH_TOKEN))((0, void, accept, (), 0, (), , noexcept, WITH_TOKEN))(     \
      (0, void, reject, (), 0, (), , noexcept, WITH_TOKEN))(                   \
      (1, double, standard_error, (), 0, (), const, noexcept, WITH_TOKEN))(    \
      (1, double, standard_error_before, (), 0, (), const, noexcept,           \
       WITH_TOKEN))(                                                           \
      (1, bool, should_reject, (), 0, (), const, noexcept, WITH_TOKEN))(       \
      (1, std::string_view, name, (), 0, (), const, noexcept, NO_TOKEN))(      \
      (1, double, computation_cost, (), 0, (), const, noexcept, WITH_TOKEN))(  \
      (0, void, set_boundary_conditions, (const BoundaryConditions &bc), 1,    \
       (bc), , , WITH_TOKEN))(                                                 \
      (0, void, set_collector, (const AtomsCollector *col), 1, (col), , ,      \
       WITH_TOKEN))(                                                           \
      (1, bool, is_rigid, (), 0, (), const, noexcept, WITH_TOKEN))(            \
      (1, bool, is_singular, (), 0, (), const, noexcept, WITH_TOKEN))(         \
      (0, void, set_n_frames, (std::size_t n), 1, (n), , noexcept,             \
       WITH_TOKEN))((0, void, set_active_frame, (std::size_t k), 1, (k), ,     \
                     noexcept, WITH_TOKEN))(                                   \
      (0, void, initialise, (), 0, (), , , WITH_TOKEN))
RMC_DEFINE_ERASED_TYPE(Constraint, RMC_CONSTRAINT_METHODS)
#undef RMC_CONSTRAINT_METHODS

// Penalty for a value outside [lo, hi]: distance to the nearest endpoint.
[[nodiscard]] FORCE_INLINE constexpr double
range_violation(double x, double lo, double hi) noexcept {
  if (x < lo) {
    return lo - x;
  }
  if (x > hi) {
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
  constexpr void
  set_boundary_conditions(Constraint::Token,
                          const BoundaryConditions &bc) noexcept {
    bc_ = &bc;
  }
  constexpr void
  set_boundary_conditions(const BoundaryConditions &bc) noexcept {
    bc_ = &bc;
  }
  // The engine's AtomsCollector (null when the collector feature is off). Lets
  // term-accumulating loops skip atoms staged/committed for removal.
  constexpr void set_collector(Constraint::Token,
                               const AtomsCollector *c) noexcept {
    collector_ = c;
  }
  constexpr void set_collector(const AtomsCollector *c) noexcept {
    collector_ = c;
  }

  void compute_before_move(Constraint::Token, const coords_t &coords,
                           std::span<const std::size_t> moved) {
    err_before_ = static_cast<Derived *>(this)->compute_error(coords, moved);
  }
  void compute_after_move(Constraint::Token, const coords_t &coords,
                          std::span<const std::size_t> moved) {
    err_after_ = static_cast<Derived *>(this)->compute_error(coords, moved);
  }
  constexpr void accept(Constraint::Token) noexcept {
    err_before_ = err_after_;
  }
  constexpr void reject(Constraint::Token) noexcept {
    // Reset err_after_ so total_error() stays consistent after short-circuited
    // compute_after_move (i.e. when this constraint was skipped).
    err_after_ = err_before_;
  }

  [[nodiscard]] constexpr double
  standard_error(Constraint::Token) const noexcept {
    return err_after_;
  }
  [[nodiscard]] constexpr double
  standard_error_before(Constraint::Token) const noexcept {
    return err_before_;
  }
  // Per-constraint downhill test. Used directly only as the RIGID hard gate
  // now; the soft accept/reject decision (incl. any uphill tolerance or
  // annealing) is made by the engine's Sampler on the TOTAL soft error.
  [[nodiscard]] constexpr bool should_reject(Constraint::Token) const noexcept {
    return err_after_ > err_before_;
  }
  // Default cost: O(1) or O(bonds). Override in O(N) / O(N²) constraints.
  [[nodiscard]] constexpr double
  computation_cost(Constraint::Token) const noexcept {
    return 1.0;
  }
  [[nodiscard]] static constexpr bool is_rigid(Constraint::Token) noexcept {
    return false;
  }
  [[nodiscard]] static constexpr bool is_singular(Constraint::Token) noexcept {
    return false;
  }
  // No-op defaults for multi-frame support; override in pair constraints.
  constexpr void set_n_frames(Constraint::Token, std::size_t) noexcept {}
  constexpr void set_active_frame(Constraint::Token, std::size_t) noexcept {}
  // No-op default: most constraints need no one-time setup. Pair constraints
  // override with the token-gated wrapper that fills their tables.
  constexpr void initialise(Constraint::Token) noexcept {}

protected:
  // True when atom `i` is staged or committed for removal — a term that
  // involves it must be skipped (contribute nothing). Cheap when the collector
  // feature is off (null pointer short-circuits before any lookup).
  [[nodiscard]] FORCE_INLINE bool absent(std::size_t i) const noexcept {
    return collector_ != nullptr && collector_->absent(i);
  }
  [[nodiscard]] FORCE_INLINE double
  distance_sq(const coords_t &c, std::size_t i, std::size_t j) const noexcept {
    vec3_t d = c.row(j).transpose() - c.row(i).transpose();
    if (bc_) {
      d = bc_min_image(*bc_, d);
    }
    return d.squaredNorm();
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
  [[nodiscard]] static constexpr double
  standard_error(Constraint::Token) noexcept {
    return 0.0;
  }
  // Rigid constraints contribute 0 to the soft total both before and after, so
  // they never enter the Sampler's ΔE — they act purely as a hard gate.
  [[nodiscard]] static constexpr double
  standard_error_before(Constraint::Token) noexcept {
    return 0.0;
  }

  [[nodiscard]] static constexpr bool is_rigid(Constraint::Token) noexcept {
    return true;
  }
};

// Only one instance of this constraint type is allowed per ConstraintCollection
// standard_error() is NOT overridden —
// singular constraints still contribute to total chi².
template <typename Derived>
class SingularConstraintBase : public ConstraintBase<Derived> {
public:
  [[nodiscard]] static constexpr bool is_singular(Constraint::Token) noexcept {
    return true;
  }
};

} // namespace RMC
