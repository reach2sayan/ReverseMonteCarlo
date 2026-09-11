#pragma once

#include <RMC/constraints/ConstraintCollection.hpp>

#include <pybind11/pybind11.h>

#include <algorithm>
#include <format>

namespace rmc::python {

namespace py = pybind11;

// ConstraintCollection::add only BOOST_ASSERTs this, which a release build
// skips: a second singular constraint of one kind would silently double-count.
inline void require_addable(const RMC::ConstraintCollection &constraints,
                            const RMC::Constraint &c) {
  if (c.is_singular() &&
      std::ranges::any_of(constraints, [&](const RMC::Constraint &held) {
        return held.name() == c.name();
      })) {
    throw py::value_error(std::format(
        "{} is singular: a collection holds at most one", c.name()));
  }
}

} // namespace rmc::python
