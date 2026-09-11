#include "errors.hpp" // register_errors

#include <pybind11/pybind11.h>

// The extension module. Nothing is bound here: each subsystem's TU binds its
// own headers, mirroring include/RMC/, and this file only fixes their order.
//
// _core is private; the public surface is the pure-Python rmc package, which
// re-exports from it and adds what a C++ binding expresses poorly (keyword-only
// options, pydantic config, protocols for Python-defined constraints).
namespace rmc::python {

void bind_core(py::module_ &m);
void bind_io(py::module_ &m);

} // namespace rmc::python

PYBIND11_MODULE(_core, m) {
  namespace rp = rmc::python;

  m.doc() = "Raw bindings for RMC. Import rmc instead.";
  m.attr("__version__") = RMC_VERSION_STRING;

  // Errors first: every other binding's unwrap() reaches for these types.
  rp::register_errors(m);

  // Then vocabulary before the things phrased in it, so later signatures name
  // AtomicStructure and BoundaryConditions rather than raw C++ types.
  rp::bind_core(m);
  rp::bind_io(m);
}
