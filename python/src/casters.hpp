#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <format>
#include <string>

// The conversions and checks that are not pybind11's job out of the box.
// Arrays, lists and optionals go through pybind11/eigen.h and pybind11/stl.h.
namespace rmc::python {

namespace py = pybind11;

// A per-atom field must have one entry per atom: constraints hold spans into
// these vectors, so a length change would leave them dangling.
inline void require_per_atom(std::size_t got, std::size_t n_atoms,
                             const char *field) {
  if (got != n_atoms) {
    throw py::value_error(std::format(
        "{} has {} entries for {} atoms; build a new AtomicStructure to change "
        "the atom count",
        field, got, n_atoms));
  }
}

// An atom index, bounds-checked (IndexError, as for a sequence).
inline std::size_t require_atom(std::size_t i, std::size_t n_atoms) {
  if (i >= n_atoms) {
    throw py::index_error(
        std::format("atom index {} out of range for {} atoms", i, n_atoms));
  }
  return i;
}

// A reader's result: a structure plus whatever cell it carries (box, bc,
// origin). One binder for LoadedStructure, VaspData, LammpsData and
// RandomStructure; members are read-only views that keep the result alive.
template <class T>
py::class_<T> bind_structure_result(py::handle scope, const char *name,
                                    const char *doc) {
  py::class_<T> cls{scope, name, doc};
  cls.def_readonly("structure", &T::structure);
  if constexpr (requires { &T::box; }) {
    cls.def_readonly("box", &T::box, "The cell matrix, read-only.");
  }
  if constexpr (requires { &T::origin; }) {
    cls.def_readonly("origin", &T::origin, "The cell's lower corner.");
  }
  if constexpr (requires { &T::bc; }) {
    cls.def_readonly("bc", &T::bc);
  }
  if constexpr (requires(const T &t) { t.periodic_bc(); }) {
    cls.def("periodic_bc", &T::periodic_bc, "PeriodicBC over box.");
  }
  return cls;
}

} // namespace rmc::python
