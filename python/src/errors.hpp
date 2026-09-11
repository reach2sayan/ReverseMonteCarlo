#pragma once

#include <RMC/core/Types.hpp> // Result<T>

#include <boost/leaf.hpp>
#include <pybind11/pybind11.h>

#include <string>
#include <type_traits>

// The one place a Result<T> becomes a Python exception. Everything else in
// this module is ordinary pybind11.
namespace rmc::python {

namespace py = pybind11;
namespace leaf = boost::leaf;

// Built once at import. By handle, not py::object: a static py::object would
// decref these interpreter-lifetime types after the GIL is already gone.
//
// RMC's errors carry a bare std::string payload, never a typed tag, so the
// Python class is chosen by the CALL SITE's subsystem, not by the payload.
struct ErrorTypes {
  py::handle base; // RmcError, which every other one derives from
  py::handle io;
  py::handle analysis;
  py::handle random_structure;
  py::handle config;
  py::handle mcsqs;
};

void register_errors(py::module_ &m);
[[nodiscard]] const ErrorTypes &error_types() noexcept;

namespace detail {
[[noreturn]] void raise(py::handle type, const char *text);
} // namespace detail

// A callable returning a Boost.LEAF result.
template <class F>
concept ResultProducer = leaf::is_result_type<std::invoke_result_t<F &>>::value;

// The success value of `make()`, or `type` raised with the error's message.
// Load-bearing, as in seitz:
//  * Takes the CALL, not its Result. LEAF's payload slots live only while a
//    matching context is active, so a Result produced beforehand arrives with
//    its message gone.
//  * Handlers run after `make()` returns, so a GIL released inside `make` is
//    held again by the time one of them raises (see unwrap_nogil).
template <ResultProducer F> auto unwrap(py::handle type, F &&make) {
  using R = std::invoke_result_t<F &>;
  using Value = typename R::value_type;
  return leaf::try_handle_all(
      [&]() -> R { return make(); },
      [&](const std::string &message) -> Value {
        detail::raise(type, message.c_str());
      },
      [&](const leaf::error_info &) -> Value {
        detail::raise(type, "rmc: unclassified failure");
      });
}

// unwrap() with the GIL released around the C++ call itself, inside the LEAF
// context. Arguments must already be C++ values: nothing Python is touched.
template <std::invocable F>
  requires ResultProducer<F>
auto unwrap_nogil(py::handle type, F &&call) {
  return unwrap(type, [&] {
    const py::gil_scoped_release unlocked;
    return call();
  });
}

} // namespace rmc::python
