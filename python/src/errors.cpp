#include "errors.hpp"

#include <array>
#include <string>

namespace rmc::python {

namespace {

ErrorTypes g_types;

// Built with the C API rather than py::exception<T>: PyErr_NewExceptionWithDoc
// makes a plain heap type that takes its docstring at creation. The reference
// is never released -- these are interpreter-lifetime type objects.
[[nodiscard]] py::handle make(py::module_ &m, const char *name, const char *doc,
                              py::handle base) {
  const std::string qualified = std::string{"rmc._core."} + name;
  const py::handle type =
      PyErr_NewExceptionWithDoc(qualified.c_str(), doc, base.ptr(), nullptr);
  if (!type) {
    throw py::error_already_set();
  }
  m.add_object(name, type);
  return type;
}

// Every subclass of RmcError: its Python name, docstring and slot.
struct Subclass {
  const char *name;
  const char *doc;
  py::handle ErrorTypes::*slot;
};
constexpr std::array<Subclass, 5> kSubclasses{{
    {"IoError", "A structure, data or checkpoint file could not be read or "
                "written.",
     &ErrorTypes::io},
    {"AnalysisError", "g(r) or ADF computation failed (bad parameters or input).",
     &ErrorTypes::analysis},
    {"RandomStructureError",
     "make_random_amorphous rejected its elements, counts or spacing.",
     &ErrorTypes::random_structure},
    {"ConfigError", "An RMCConfig could not be turned into an engine.",
     &ErrorTypes::config},
    {"McsqsError", "The SQS lattice/cluster pipeline failed.",
     &ErrorTypes::mcsqs},
}};

} // namespace

namespace detail {

void raise(py::handle type, const char *text) {
  py::set_error(type, text);
  throw py::error_already_set();
}

} // namespace detail

const ErrorTypes &error_types() noexcept { return g_types; }

void register_errors(py::module_ &m) {
  g_types.base = make(m, "RmcError",
                      "Base of every error this library reports. The C++ side "
                      "returns Result<T> and never throws; this is what that "
                      "becomes.",
                      PyExc_Exception);
  for (const Subclass &s : kSubclasses) {
    g_types.*s.slot = make(m, s.name, s.doc, g_types.base);
  }
}

} // namespace rmc::python
