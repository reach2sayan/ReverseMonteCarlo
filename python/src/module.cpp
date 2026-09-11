#include <pybind11/pybind11.h>

// The extension module. Nothing is bound here: each subsystem's TU binds its
// own headers, mirroring include/RMC/, and this file only fixes their order.
//
// _core is private; the public surface is the pure-Python rmc package, which
// re-exports from it and adds what a C++ binding expresses poorly (keyword-only
// options, pydantic config, protocols for Python-defined constraints).
namespace rmc::python {
namespace py = pybind11;
} // namespace rmc::python

PYBIND11_MODULE(_core, m) {
  m.doc() = "Raw bindings for RMC. Import rmc instead.";
  m.attr("__version__") = RMC_VERSION_STRING;
}
