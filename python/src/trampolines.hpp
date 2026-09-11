#pragma once

#include <RMC/constraints/Constraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>
#include <RMC/generators/MoveGenerator.hpp>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

// Constraints and move generators written in Python, adapted to the C++
// concepts so the engine drives them like any other.
//
// Every call into Python acquires the GIL (RAII): Engine.run releases it, and
// ensembles call from worker threads. The held py::object is copied or
// destroyed only on paths Python starts (add_constraint, Group construction,
// engine teardown), all of which hold the GIL.
namespace rmc::python {

namespace py = pybind11;

// Indices as a fresh (k,) uint64 array: a copy, because the span borrows the
// engine's state for the duration of one call.
[[nodiscard]] inline py::array_t<std::size_t>
index_array(std::span<const std::size_t> indices) {
  return py::array_t<std::size_t>(static_cast<py::ssize_t>(indices.size()),
                                   indices.data());
}

// An optional attribute read once, at wrap time.
template <class T>
[[nodiscard]] T attr_or(const py::object &o, const char *name, T fallback) {
  return py::hasattr(o, name) ? o.attr(name).cast<T>() : fallback;
}

// A Python object with compute_error(coords, moved) -> float and, optionally,
// `rigid`, `cost`, `name`, set_boundary_conditions(bc) and initialise().
class PyConstraint : public RMC::ConstraintBase<PyConstraint> {
public:
  explicit PyConstraint(py::object impl) : impl_{std::move(impl)} {
    if (!py::hasattr(impl_, "compute_error")) {
      throw py::type_error(
          "a Python constraint needs compute_error(coords, moved) -> float");
    }
    name_ = attr_or<std::string>(
        impl_, "name", py::type::of(impl_).attr("__name__").cast<std::string>());
    cost_ = attr_or(impl_, "cost", 1.0);
    rigid_ = attr_or(impl_, "rigid", false);
    has_set_bc_ = py::hasattr(impl_, "set_boundary_conditions");
    has_initialise_ = py::hasattr(impl_, "initialise");
  }

  // coords arrives as a read-only zero-copy view, valid for this call only.
  [[nodiscard]] double compute_error(const RMC::coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    const py::gil_scoped_acquire gil;
    return impl_
        .attr("compute_error")(py::cast(coords, py::return_value_policy::reference),
                               index_array(moved))
        .cast<double>();
  }

  [[nodiscard]] std::string_view name() const noexcept { return name_; }
  [[nodiscard]] double computation_cost() const noexcept { return cost_; }
  [[nodiscard]] bool is_rigid() const noexcept { return rigid_; }
  // A rigid constraint is a hard gate and adds nothing to the total chi^2.
  [[nodiscard]] double standard_error() const noexcept {
    return rigid_ ? 0.0 : ConstraintBase::standard_error();
  }
  [[nodiscard]] double standard_error_before() const noexcept {
    return rigid_ ? 0.0 : ConstraintBase::standard_error_before();
  }

  void set_boundary_conditions(const RMC::BoundaryConditions &bc) {
    ConstraintBase::set_boundary_conditions(bc);
    if (has_set_bc_) {
      const py::gil_scoped_acquire gil;
      impl_.attr("set_boundary_conditions")(
          py::cast(bc, py::return_value_policy::reference));
    }
  }
  void initialise() {
    if (has_initialise_) {
      const py::gil_scoped_acquire gil;
      impl_.attr("initialise")();
    }
  }

  [[nodiscard]] const py::object &impl() const noexcept { return impl_; }

private:
  py::object impl_;
  std::string name_;
  double cost_{1.0};
  bool rigid_{false};
  bool has_set_bc_{false};
  bool has_initialise_{false};
};
static_assert(RMC::CConstraint<PyConstraint>);

// A Python object with generate(coords, indices) -> None that moves atoms by
// writing into coords, and optionally `modifies_species` and
// rejection_override() -> bool | None.
class PyMoveGenerator : public RMC::MoveGeneratorBase<PyMoveGenerator> {
public:
  explicit PyMoveGenerator(py::object impl) : impl_{std::move(impl)} {
    if (!py::hasattr(impl_, "generate")) {
      throw py::type_error(
          "a Python move generator needs generate(coords, indices) -> None");
    }
    modifies_species_ = attr_or(impl_, "modifies_species", false);
    has_rejection_override_ = py::hasattr(impl_, "rejection_override");
  }

  // coords arrives as a writable zero-copy view of the frame being moved.
  void generate(RMC::coords_t &coords, std::span<const std::size_t> indices) {
    const py::gil_scoped_acquire gil;
    impl_.attr("generate")(py::cast(coords, py::return_value_policy::reference),
                           index_array(indices));
  }

  [[nodiscard]] std::optional<bool> rejection_override() const {
    if (!has_rejection_override_) {
      return std::nullopt;
    }
    const py::gil_scoped_acquire gil;
    return impl_.attr("rejection_override")().cast<std::optional<bool>>();
  }
  [[nodiscard]] bool modifies_species() const noexcept {
    return modifies_species_;
  }

  [[nodiscard]] const py::object &impl() const noexcept { return impl_; }

private:
  py::object impl_;
  bool modifies_species_{false};
  bool has_rejection_override_{false};
};
static_assert(RMC::CMoveGenerator<PyMoveGenerator>);

} // namespace rmc::python
