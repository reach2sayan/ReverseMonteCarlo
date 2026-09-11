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
#include <variant>

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

// Calls f(std::type_identity<T>{}) for each T in List (an mp11 list), so the
// types need not be default-constructible.
template <class List, class F> void for_each_type(F &&f) {
  boost::mp11::mp_for_each<boost::mp11::mp_transform<std::type_identity, List>>(
      std::forward<F>(f));
}

namespace detail {
template <class... Ts> std::variant<Ts...> as_variant(const std::variant<Ts...> &);
} // namespace detail

// The alternatives of V -- a std::variant, or a type deriving from one such as
// Sampler and GroupSelector -- as an mp11 list. Overload sets generated from it
// grow with the variant: a new selector in C++ is a new overload in Python.
template <class V>
using Alternatives =
    boost::mp11::mp_rename<decltype(detail::as_variant(std::declval<const V &>())),
                           boost::mp11::mp_list>;

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

// Field equality that is safe for Eigen matrices (their == asserts equal
// shapes, which a release build does not check) and optionals of them.
template <class F> [[nodiscard]] bool field_equal(const F &a, const F &b) {
  if constexpr (requires { a.rows(); a.cols(); }) {
    return a.rows() == b.rows() && a.cols() == b.cols() && a == b;
  } else if constexpr (requires { a.has_value(); *a; }) {
    return a.has_value() == b.has_value() && (!a || field_equal(*a, *b));
  } else {
    return a == b;
  }
}

template <class T, template <class...> class L, class... D>
[[nodiscard]] bool equal(const T &a, const T &b, L<D...>) {
  return (field_equal(a.*D::pointer, b.*D::pointer) && ...);
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
