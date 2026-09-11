#pragma once

#include <boost/describe/enumerators.hpp>
#include <boost/describe/members.hpp>
#include <boost/mp11/algorithm.hpp>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

// Binding from Boost.Describe metadata, so adding an enumerator or a field in
// C++ adds it in Python with no second list to keep in step.
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

template <class T>
concept DescribedStruct = boost::describe::has_describe_members<T>::value &&
                          std::default_initializable<T>;

namespace detail {

template <class T>
using PublicMembers =
    boost::describe::describe_members<T, boost::describe::mod_public>;

template <class T, class D>
using FieldType =
    std::remove_cvref_t<decltype(std::declval<T &>().*D::pointer)>;

// T(*, field=default, ...) with one keyword per described field; defaults are
// T{}'s, so the Python signature documents the C++ defaults exactly.
template <class T, template <class...> class L, class... D>
void def_init(py::class_<T> &cls, L<D...>) {
  const T defaults{};
  cls.def(py::init([](FieldType<T, D>... values) {
            T t{};
            ((t.*D::pointer = std::move(values)), ...);
            return t;
          }),
          py::kw_only(), (py::arg(D::name) = defaults.*D::pointer)...);
}

template <class T, template <class...> class L, class... D>
[[nodiscard]] bool equal(const T &a, const T &b, L<D...>) {
  return ((a.*D::pointer == b.*D::pointer) && ...);
}

} // namespace detail

// A plain struct as a Python class: a keyword-only constructor, read/write
// fields, __repr__ from each field's Python repr, and __eq__ when every field
// compares. Returns the class so callers can add methods.
template <DescribedStruct T>
py::class_<T> describe_struct(py::handle scope, const char *name,
                              const char *doc) {
  using Members = detail::PublicMembers<T>;
  py::class_<T> cls{scope, name, doc};
  detail::def_init<T>(cls, Members{});
  boost::mp11::mp_for_each<Members>(
      [&](auto d) { cls.def_readwrite(d.name, d.pointer); });

  cls.def("__repr__", [name = std::string{name}](const T &t) {
    std::string out = name + "(";
    const char *sep = "";
    boost::mp11::mp_for_each<Members>([&](auto d) {
      out += std::string{sep} + d.name + "=" +
             py::repr(py::cast(t.*d.pointer)).template cast<std::string>();
      sep = ", ";
    });
    return out + ")";
  });

  if constexpr (requires(const T &a) {
                  { detail::equal(a, a, Members{}) } -> std::same_as<bool>;
                }) {
    cls.def(
        "__eq__",
        [](const T &a, const T &b) { return detail::equal(a, b, Members{}); },
        py::is_operator());
  }
  return cls;
}

} // namespace rmc::python
