#include "describe.hpp"

#include <RMC/sampling/Sampler.hpp>

#include <boost/describe/class.hpp>

#include <pybind11/pybind11.h>

#include <cstdint>

namespace RMC {
// Described here rather than in AnnealingSampler.hpp: only the binding reads it.
BOOST_DESCRIBE_STRUCT(AnnealingSampler::Schedule, (),
                      (t0, cooling, interval, t_min))
} // namespace RMC

// Acceptance policies: every alternative of RMC::Sampler.
namespace rmc::python {

namespace {

using RMC::AnnealingSampler;
using RMC::GreedySampler;
using RMC::MetropolisSampler;

// What every sampler shares: the acceptance test, callable outside an engine.
template <class T>
py::class_<T> bind_sampler(py::module_ &m, const char *name, const char *doc) {
  py::class_<T> cls{m, name, doc};
  cls.def(
      "accept",
      [](const T &s, double e_before, double e_after, std::uint64_t step,
         double u01) { return s.accept(e_before, e_after, step, u01); },
      py::arg("e_before"), py::arg("e_after"), py::arg("step") = 0,
      py::arg("u01") = 0.5,
      "Whether a move from e_before to e_after is accepted at step, given a "
      "uniform draw u01 in [0, 1).");
  return cls;
}

} // namespace

void bind_sampling(py::module_ &m) {
  bind_sampler<GreedySampler>(
      m, "GreedySampler",
      "Accept downhill moves, and uphill ones within tolerance (the engine's "
      "default).")
      .def(py::init<double>(), py::arg("tolerance") = 0.0)
      .def_property_readonly("tolerance", &GreedySampler::tolerance);

  bind_sampler<MetropolisSampler>(
      m, "MetropolisSampler",
      "Accept uphill moves with probability exp(-dE / T); T <= 0 is greedy.")
      .def(py::init<double>(), py::arg("temperature"))
      .def_property_readonly("temperature", &MetropolisSampler::temperature);

  auto annealing = bind_sampler<AnnealingSampler>(
      m, "AnnealingSampler",
      "Metropolis at T = max(t_min, t0 * cooling ** (step // interval)).");
  describe_struct<AnnealingSampler::Schedule>(
      annealing, "Schedule",
      "Geometric cooling: T starts at t0 and is multiplied by cooling every "
      "interval steps, never below t_min.");
  annealing
      .def(py::init<AnnealingSampler::Schedule>(),
           py::arg("schedule") = AnnealingSampler::Schedule{})
      .def("temperature", &AnnealingSampler::temperature, py::arg("step"),
           "The temperature at step.");
}

} // namespace rmc::python
