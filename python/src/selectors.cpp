#include "describe.hpp"

#include <RMC/selectors/GroupSelector.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

// Group-selection policies: every alternative of RMC::GroupSelector.
namespace rmc::python {

namespace {

using RMC::GroupSelector;
using RMC::RecursiveGroupSelector;
using RMC::RecursiveMode;

// What every selector shares: the next index, and the outcome feedback for
// the adaptive ones.
template <class T>
py::class_<T> bind_selector(py::module_ &m, const char *name, const char *doc) {
  py::class_<T> cls{m, name, doc};
  cls.def(
      "select",
      [](T &s, std::size_t n_groups) {
        if (n_groups == 0) {
          throw py::value_error("select needs n_groups >= 1");
        }
        return s.select(n_groups);
      },
      py::arg("n_groups"), "The next group index, in [0, n_groups).");
  if constexpr (requires(T &s) { s.feedback(std::size_t{0}, true); }) {
    cls.def(
        "feedback",
        [](T &s, std::size_t group, bool accepted) { s.feedback(group, accepted); },
        py::arg("group"), py::arg("accepted"),
        "Report a move's outcome; the engine does this after every trial.");
  }
  return cls;
}

} // namespace

void bind_selectors(py::module_ &m) {
  using RMC::DirectionalOrderSelector;
  using RMC::OrderedSelector;
  using RMC::RandomSelector;
  using RMC::SmartRandomSelector;
  using RMC::WeightedRandomSelector;
  constexpr std::uint32_t kSeed = 42;

  bind_selector<RandomSelector>(m, "RandomSelector",
                                "A uniformly random group (the engine's default).")
      .def(py::init<std::uint32_t>(), py::kw_only(), py::arg("seed") = kSeed);

  bind_selector<WeightedRandomSelector>(
      m, "WeightedRandomSelector",
      "Groups drawn with probability proportional to weights (uniform when the "
      "group count differs from len(weights)).")
      .def(py::init<std::vector<double>, std::uint32_t>(), py::arg("weights"),
           py::kw_only(), py::arg("seed") = kSeed)
      .def_property(
          "weights", [](const WeightedRandomSelector &s) { return s.weights; },
          [](WeightedRandomSelector &s, const std::vector<double> &w) {
            s.set_weights(w);
          });

  bind_selector<OrderedSelector>(m, "OrderedSelector", "Groups in order, cycling.")
      .def(py::init<>())
      .def_readwrite("current", &OrderedSelector::current);

  bind_selector<SmartRandomSelector>(
      m, "SmartRandomSelector",
      "Adaptive: an accepted move multiplies its group's weight by "
      "bias_factor, a rejected one divides it.")
      .def(py::init<double, std::uint32_t>(), py::arg("bias_factor") = 1.1,
           py::kw_only(), py::arg("seed") = kSeed)
      .def_readwrite("bias_factor", &SmartRandomSelector::bias_factor)
      .def_property_readonly("weights", &SmartRandomSelector::weights,
                             "Normalised selection weights.");

  bind_selector<DirectionalOrderSelector>(
      m, "DirectionalOrderSelector",
      "Groups ordered by centroid distance from reference, nearest (or "
      "farthest) first, cycling.")
      .def(py::init<const RMC::vec3_t &, const std::vector<RMC::vec3_t> &, bool>(),
           py::arg("reference"), py::arg("centroids"), py::kw_only(),
           py::arg("nearest_first") = true)
      .def_readonly("nearest_first", &DirectionalOrderSelector::nearest_first);

  auto recursive = bind_selector<RecursiveGroupSelector>(
      m, "RecursiveGroupSelector",
      "Wraps another selector and repeats a group up to max_retries times "
      "after an accepted (Refine) or rejected (Explore) move.");
  for_each_type<Alternatives<GroupSelector>>([&](auto id) {
    using Inner = typename decltype(id)::type;
    recursive.def(py::init([](const Inner &inner, RecursiveMode mode, int retries) {
                    return RecursiveGroupSelector{GroupSelector{inner}, mode, retries};
                  }),
                  py::arg("inner"), py::arg("mode") = RecursiveMode::Refine,
                  py::kw_only(), py::arg("max_retries") = 5);
  });
  recursive.def_readwrite("mode", &RecursiveGroupSelector::mode)
      .def_readwrite("max_retries", &RecursiveGroupSelector::max_retries);
}

} // namespace rmc::python
