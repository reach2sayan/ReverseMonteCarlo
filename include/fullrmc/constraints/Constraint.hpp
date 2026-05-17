#pragma once
#include <cmath>
#include <fullrmc/core/BoundaryConditions.hpp>
#include <fullrmc/core/Types.hpp>
#include <memory>
#include <string>

namespace fullrmc {

class IConstraint; // forward for friend declaration

namespace detail {
struct ConstraintToken {
private:
  constexpr ConstraintToken() = default;
  friend class ::fullrmc::IConstraint;
};
} // namespace detail

// Satisfied by any type that implements the passkey-gated constraint interface.
template <typename T>
concept CConstraint =
    requires(T &c, detail::ConstraintToken tok, const coords_t &coords,
             std::span<const std::size_t> moved, const BoundaryConditions &bc) {
      c.compute_before_move(tok, coords, moved);
      c.compute_after_move(tok, coords, moved);
      c.accept(tok);
      c.reject(tok);
      { c.standard_error(tok) } -> std::convertible_to<double>;
      { c.should_reject(tok) } -> std::convertible_to<bool>;
      { c.name() } -> std::convertible_to<std::string>;
      c.set_boundary_conditions(tok, bc);
    };

class IConstraint {
public:
  using Token = detail::ConstraintToken;

  template <CConstraint T>
  IConstraint(T x)
      : self_(std::make_unique<ConstraintModel<T>>(std::move(x))) {}
  IConstraint(const IConstraint &s) : self_{s.self_->clone()} {}
  IConstraint(IConstraint &&s) noexcept : self_{std::move(s.self_)} {}
  IConstraint &operator=(const IConstraint &s) {
    self_ = s.self_->clone();
    return *this;
  }
  IConstraint &operator=(IConstraint &&s) noexcept {
    self_ = std::move(s.self_);
    return *this;
  }

  void compute_before_move(const coords_t &coords,
                           std::span<const std::size_t> moved) {
    self_->compute_before_move(coords, moved);
  }
  void compute_after_move(const coords_t &coords,
                          std::span<const std::size_t> moved) {
    self_->compute_after_move(coords, moved);
  }
  void accept() noexcept { self_->accept(); }
  void reject() noexcept { self_->reject(); }
  [[nodiscard]] double standard_error() const noexcept {
    return self_->standard_error();
  }
  [[nodiscard]] bool should_reject() const noexcept {
    return self_->should_reject();
  }
  [[nodiscard]] std::string name() const { return self_->name(); }
  void set_boundary_conditions(const BoundaryConditions &bc) {
    self_->set_boundary_conditions(bc);
  }

private:
  static Token make_token() noexcept { return {}; }

  struct ConstraintConcept {
    virtual ~ConstraintConcept() = default;
    virtual void compute_before_move(const coords_t &,
                                     std::span<const std::size_t>) = 0;
    virtual void compute_after_move(const coords_t &,
                                    std::span<const std::size_t>) = 0;
    virtual void accept() noexcept = 0;
    virtual void reject() noexcept = 0;
    [[nodiscard]] virtual double standard_error() const noexcept = 0;
    [[nodiscard]] virtual bool should_reject() const noexcept = 0;
    [[nodiscard]] virtual std::string name() const = 0;
    virtual void set_boundary_conditions(const BoundaryConditions &) = 0;
    virtual std::unique_ptr<ConstraintConcept> clone() const = 0;
  };

  template <CConstraint T> struct ConstraintModel final : ConstraintConcept {
    explicit ConstraintModel(T x) : data_(std::move(x)) {}
    void compute_before_move(const coords_t &c,
                             std::span<const std::size_t> m) override {
      data_.compute_before_move(make_token(), c, m);
    }
    void compute_after_move(const coords_t &c,
                            std::span<const std::size_t> m) override {
      data_.compute_after_move(make_token(), c, m);
    }
    void accept() noexcept override { data_.accept(make_token()); }
    void reject() noexcept override { data_.reject(make_token()); }
    double standard_error() const noexcept override {
      return data_.standard_error(make_token());
    }
    bool should_reject() const noexcept override {
      return data_.should_reject(make_token());
    }
    std::string name() const override { return data_.name(); }
    void set_boundary_conditions(const BoundaryConditions &bc) override {
      data_.set_boundary_conditions(make_token(), bc);
    }
    std::unique_ptr<ConstraintConcept> clone() const override {
      return std::make_unique<ConstraintModel<T>>(data_);
    }
    T data_;
  };

  std::unique_ptr<ConstraintConcept> self_;
};

// CRTP mixin — inherit to get all token-gated methods implemented via a single
// compute_error(coords, moved) -> double that Derived must provide.
// Derived must also provide: std::string name() const
template <typename Derived> class ConstraintBase {
protected:
  const BoundaryConditions *bc_{nullptr};
  double err_before_{0.0};
  double err_after_{0.0};

public:
  bool flexible = false;
  double tolerance = 0.0;

  void set_boundary_conditions(IConstraint::Token,
                               const BoundaryConditions &bc) noexcept {
    bc_ = &bc;
  }
  // Non-token overload for pre-wrap configuration.
  void set_boundary_conditions(const BoundaryConditions &bc) noexcept {
    bc_ = &bc;
  }

  void compute_before_move(IConstraint::Token, const coords_t &coords,
                           std::span<const std::size_t> moved) {
    err_before_ = static_cast<Derived *>(this)->compute_error(coords, moved);
  }
  void compute_after_move(IConstraint::Token, const coords_t &coords,
                          std::span<const std::size_t> moved) {
    err_after_ = static_cast<Derived *>(this)->compute_error(coords, moved);
  }
  void accept(IConstraint::Token) noexcept { err_before_ = err_after_; }
  void reject(IConstraint::Token) noexcept {}

  [[nodiscard]] double standard_error(IConstraint::Token) const noexcept {
    return err_after_;
  }
  [[nodiscard]] bool should_reject(IConstraint::Token) const noexcept {
    return flexible ? (err_after_ > err_before_ + tolerance)
                    : (err_after_ > err_before_);
  }

protected:
  [[nodiscard]] double distance_sq(const coords_t &c, std::size_t i,
                                   std::size_t j) const noexcept {
    vec3_t d = c.row(j).transpose() - c.row(i).transpose();
    if (bc_)
      d = bc_min_image(*bc_, d);
    return d.squaredNorm();
  }
  [[nodiscard]] double distance(const coords_t &c, std::size_t i,
                                std::size_t j) const noexcept {
    return std::sqrt(distance_sq(c, i, j));
  }
};

} // namespace fullrmc
