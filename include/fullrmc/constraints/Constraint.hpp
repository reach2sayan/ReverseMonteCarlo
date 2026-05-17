#pragma once
#include <fullrmc/core/Types.hpp>
#include <fullrmc/core/BoundaryConditions.hpp>
#include <memory>
#include <string>

namespace fullrmc {

// ---- Type-erased constraint interface ----
struct IConstraint {
    virtual ~IConstraint() = default;

    virtual void compute_before_move(const coords_t& coords,
                                      std::span<const index_t> moved) = 0;
    virtual void compute_after_move(const coords_t& coords,
                                     std::span<const index_t> moved) = 0;
    virtual void accept() noexcept = 0;
    virtual void reject() noexcept = 0;

    [[nodiscard]] virtual real_t standard_error()  const noexcept = 0;
    [[nodiscard]] virtual bool   should_reject()   const noexcept = 0;
    [[nodiscard]] virtual std::string name()       const = 0;

    // Flexible constraints tolerate a worsening up to `tolerance`.
    bool flexible  = false;
    real_t tolerance = 0.0;
};

// ---- CRTP base ---- avoids virtual call in the hot path while providing the
// vtable for storage in std::vector<std::unique_ptr<IConstraint>>.
template<typename Derived>
class ConstraintBase : public IConstraint {
protected:
    const BoundaryConditions* bc_{nullptr};
    real_t err_before_{0.0};
    real_t err_after_ {0.0};

public:
    void set_boundary_conditions(const BoundaryConditions& bc) noexcept {
        bc_ = &bc;
    }

    void compute_before_move(const coords_t& coords,
                              std::span<const index_t> moved) final {
        err_before_ = static_cast<Derived*>(this)->compute_error(coords, moved);
    }
    void compute_after_move(const coords_t& coords,
                             std::span<const index_t> moved) final {
        err_after_ = static_cast<Derived*>(this)->compute_error(coords, moved);
    }
    void accept() noexcept final { err_before_ = err_after_; }
    void reject() noexcept final { /* err_before_ already valid */ }

    [[nodiscard]] real_t standard_error() const noexcept final { return err_after_; }
    [[nodiscard]] bool   should_reject()  const noexcept final {
        if (flexible) return err_after_ > err_before_ + tolerance;
        return err_after_ > err_before_;
    }

protected:
    // Helpers for subclasses.
    [[nodiscard]] real_t distance_sq(const coords_t& c,
                                      index_t i, index_t j) const noexcept {
        vec3_t d = c.row(j).transpose() - c.row(i).transpose();
        if (bc_) d = bc_min_image(*bc_, d);
        return d.squaredNorm();
    }
    [[nodiscard]] real_t distance(const coords_t& c,
                                   index_t i, index_t j) const noexcept {
        return std::sqrt(distance_sq(c, i, j));
    }
};

} // namespace fullrmc
