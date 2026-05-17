#pragma once
#include <fullrmc/constraints/Constraint.hpp>
#include <vector>
#include <memory>
#include <string>

namespace fullrmc {

// Composite: holds any number of IConstraint instances and aggregates them.
// Evaluation short-circuits on the first rigid constraint that would reject,
// but continues for all flexible constraints (they only accumulate err_after_).
class ConstraintCollection {
public:
    void add(std::unique_ptr<IConstraint> c) {
        constraints_.push_back(std::move(c));
    }

    template<typename C, typename... Args>
    C& emplace(Args&&... args) {
        auto ptr = std::make_unique<C>(std::forward<Args>(args)...);
        C& ref = *ptr;
        constraints_.push_back(std::move(ptr));
        return ref;
    }

    void set_boundary_conditions(const BoundaryConditions& bc) noexcept {
        bc_ = &bc;
    }

    void compute_before_move(const coords_t& coords,
                              std::span<const index_t> moved) {
        for (auto& c : constraints_)
            c->compute_before_move(coords, moved);
    }

    void compute_after_move(const coords_t& coords,
                             std::span<const index_t> moved) {
        for (auto& c : constraints_)
            c->compute_after_move(coords, moved);
    }

    // Returns true if any constraint votes to reject.
    [[nodiscard]] bool should_reject() const noexcept {
        for (auto& c : constraints_)
            if (c->should_reject()) return true;
        return false;
    }

    void accept() noexcept {
        for (auto& c : constraints_) c->accept();
    }

    void reject() noexcept {
        for (auto& c : constraints_) c->reject();
    }

    [[nodiscard]] real_t total_error() const noexcept {
        real_t total = 0.0;
        for (auto& c : constraints_) total += c->standard_error();
        return total;
    }

    [[nodiscard]] std::size_t size() const noexcept { return constraints_.size(); }

    [[nodiscard]] IConstraint& operator[](std::size_t i) { return *constraints_[i]; }
    [[nodiscard]] const IConstraint& operator[](std::size_t i) const {
        return *constraints_[i];
    }

private:
    std::vector<std::unique_ptr<IConstraint>> constraints_;
    const BoundaryConditions* bc_{nullptr};
};

} // namespace fullrmc
