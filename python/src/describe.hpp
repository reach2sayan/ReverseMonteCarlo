#pragma once

#include <boost/describe/enumerators.hpp>
#include <boost/mp11/algorithm.hpp>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>

// Binding from Boost.Describe metadata, so adding an enumerator (or a field)
// in C++ adds it in Python with no second list to keep in step.
namespace rmc::python {

namespace py = pybind11;

template <class E>
concept DescribedEnum = boost::describe::has_describe_enumerators<E>::value;

// A Python enum.IntEnum whose members are E's enumerators, C++ names kept.
template <DescribedEnum E>
void describe_enum(py::handle scope, const char *name, const char *doc) {
  py::native_enum<E> e{scope, name, "enum.IntEnum", doc};
  boost::mp11::mp_for_each<boost::describe::describe_enumerators<E>>(
      [&](auto d) { e.value(d.name, d.value); });
  e.finalize();
}

} // namespace rmc::python
