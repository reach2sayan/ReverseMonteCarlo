#pragma once
#include <RMC/core/BoundaryConditions.hpp>
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
             std::size_t k) {
      c.compute_before_move(tok, coords, moved);
      c.compute_after_move(tok, coords, moved);
      c.accept(tok);
      c.reject(tok);
      { c.standard_error(tok) } -> std::convertible_to<double>;
      { c.should_reject(tok) } -> std::convertible_to<bool>;
      { c.name() } -> std::convertible_to<std::string_view>;
      { c.computation_cost(tok) } -> std::convertible_to<double>;
      c.set_boundary_conditions(tok, bc);
      { c.is_rigid(tok) } -> std::convertible_to<bool>;
      { c.is_singular(tok) } -> std::convertible_to<bool>;
      c.set_n_frames(tok, k);
      c.set_active_frame(tok, k);
    };

class Constraint {
public:
  using Token = detail::ConstraintToken;

  template <CConstraint T>
  constexpr Constraint(T x)
      : self_(std::make_unique<ConstraintModel<T>>(std::move(x))) {}
  constexpr Constraint(const Constraint &s) : self_{s.self_->clone()} {}
  constexpr Constraint(Constraint &&s) noexcept : self_{std::move(s.self_)} {}
  constexpr Constraint &operator=(const Constraint &s) {
    self_ = s.self_->clone();
    return *this;
  }
  constexpr Constraint &operator=(Constraint &&s) noexcept {
    self_ = std::move(s.self_);
    return *this;
  }

  constexpr void compute_before_move(const coords_t &coords,
                                     std::span<const std::size_t> moved) {
    self_->compute_before_move(coords, moved);
  }
  constexpr void compute_after_move(const coords_t &coords,
                                    std::span<const std::size_t> moved) {
    self_->compute_after_move(coords, moved);
  }
  constexpr void accept() noexcept { self_->accept(); }
  constexpr void reject() noexcept { self_->reject(); }
  [[nodiscard]] constexpr double standard_error() const noexcept {
    return self_->standard_error();
  }
  [[nodiscard]] constexpr bool should_reject() const noexcept {
    return self_->should_reject();
  }
  [[nodiscard]] constexpr std::string_view name() const noexcept {
    return self_->name();
  }
  [[nodiscard]] constexpr double computation_cost() const noexcept {
    return self_->computation_cost();
  }
  constexpr void set_boundary_conditions(const BoundaryConditions &bc) {
    self_->set_boundary_conditions(bc);
  }
  [[nodiscard]] constexpr bool is_rigid() const noexcept {
    return self_->is_rigid();
  }
  [[nodiscard]] constexpr bool is_singular() const noexcept {
    return self_->is_singular();
  }
  constexpr void set_n_frames(std::size_t n) noexcept {
    self_->set_n_frames(n);
  }
  constexpr void set_active_frame(std::size_t k) noexcept {
    self_->set_active_frame(k);
  }

private:
  static Token make_token() noexcept { return {}; }
  struct ConstraintConcept {
    virtual ~ConstraintConcept() = default;
    virtual void compute_before_move(const coords_t &,
                                     std::span<const std::size_t>) = 0;
    virtual void compute_after_move(const coords_t &,
                                    std::span<const std::size_t>) = 0;
    virtual constexpr void accept() noexcept = 0;
    virtual constexpr void reject() noexcept = 0;
    [[nodiscard]] virtual double standard_error() const noexcept = 0;
    [[nodiscard]] virtual bool should_reject() const noexcept = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual double computation_cost() const noexcept = 0;
    virtual void set_boundary_conditions(const BoundaryConditions &) = 0;
    [[nodiscard]] virtual bool is_rigid() const noexcept = 0;
    [[nodiscard]] virtual bool is_singular() const noexcept = 0;
    virtual void set_n_frames(std::size_t) noexcept = 0;
    virtual void set_active_frame(std::size_t) noexcept = 0;
    virtual std::unique_ptr<ConstraintConcept> clone() const = 0;
  };

  template <CConstraint T> struct ConstraintModel final : ConstraintConcept {
    constexpr explicit ConstraintModel(T x) : data_(std::move(x)) {}
    constexpr void
    compute_before_move(const coords_t &c,
                        std::span<const std::size_t> m) override {
      data_.compute_before_move(make_token(), c, m);
    }
    constexpr void compute_after_move(const coords_t &c,
                                      std::span<const std::size_t> m) override {
      data_.compute_after_move(make_token(), c, m);
    }
    constexpr void accept() noexcept override { data_.accept(make_token()); }
    constexpr void reject() noexcept override { data_.reject(make_token()); }
    constexpr double standard_error() const noexcept override {
      return data_.standard_error(make_token());
    }
    constexpr bool should_reject() const noexcept override {
      return data_.should_reject(make_token());
    }
    constexpr std::string_view name() const noexcept override {
      return data_.name();
    }
    constexpr double computation_cost() const noexcept override {
      return data_.computation_cost(make_token());
    }
    constexpr void
    set_boundary_conditions(const BoundaryConditions &bc) override {
      data_.set_boundary_conditions(make_token(), bc);
    }
    constexpr bool is_rigid() const noexcept override {
      return data_.is_rigid(make_token());
    }
    constexpr bool is_singular() const noexcept override {
      return data_.is_singular(make_token());
    }
    void set_n_frames(std::size_t n) noexcept override {
      data_.set_n_frames(make_token(), n);
    }
    void set_active_frame(std::size_t k) noexcept override {
      data_.set_active_frame(make_token(), k);
    }
    constexpr std::unique_ptr<ConstraintConcept> clone() const override {
      return std::make_unique<ConstraintModel<T>>(data_);
    }
    T data_;
  };

  std::unique_ptr<ConstraintConcept> self_;
};

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
  double err_before_{0.0};
  double err_after_{0.0};

public:
  bool flexible = false;
  double tolerance = 0.0;

  constexpr void
  set_boundary_conditions(Constraint::Token,
                          const BoundaryConditions &bc) noexcept {
    bc_ = &bc;
  }
  constexpr void
  set_boundary_conditions(const BoundaryConditions &bc) noexcept {
    bc_ = &bc;
  }

  constexpr void compute_before_move(Constraint::Token, const coords_t &coords,
                                     std::span<const std::size_t> moved) {
    err_before_ = static_cast<Derived *>(this)->compute_error(coords, moved);
  }
  constexpr void compute_after_move(Constraint::Token, const coords_t &coords,
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
  [[nodiscard]] constexpr bool
  should_reject(Constraint::Token) const noexcept {
    return flexible ? (err_after_ > err_before_ + tolerance)
                    : (err_after_ > err_before_);
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

protected:
  [[nodiscard]] constexpr FORCE_INLINE double
  distance_sq(const coords_t &c, std::size_t i, std::size_t j) const noexcept {
    vec3_t d = c.row(j).transpose() - c.row(i).transpose();
    if (bc_) {
      d = bc_min_image(*bc_, d);
    }
    return d.squaredNorm();
  }
  [[nodiscard]] constexpr FORCE_INLINE double
  distance(const coords_t &c, std::size_t i, std::size_t j) const noexcept {
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
