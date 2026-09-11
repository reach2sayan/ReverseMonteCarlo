#include "casters.hpp"
#include "constraints.hpp"
#include "describe.hpp"

#include <RMC/Engine.hpp>
#include <RMC/callbacks/Chi2CollectorCallback.hpp>
#include <RMC/callbacks/HistogramCallback.hpp>
#include <RMC/callbacks/PDBSnapshotCallback.hpp>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <cstdint>
#include <filesystem>
#include <format>
#include <utility>

// The engine and its step callbacks.
namespace rmc::python {

namespace {

using RMC::AtomicStructure;
using RMC::BoundaryConditions;
using RMC::Constraint;
using RMC::Engine;

// Engine::run only BOOST_ASSERTs this, which a release build skips.
void require_groups(const Engine &e) {
  if (e.n_groups() == 0) {
    throw py::value_error("the engine has no groups: call build_atomic_groups() "
                          "or add_group() first");
  }
}

std::uint64_t require_positive(std::uint64_t n, const char *what) {
  if (n == 0) {
    throw py::value_error(std::format("{} must be >= 1", what));
  }
  return n;
}

void bind_engine_class(py::module_ &m) {
  py::class_<Engine> cls{
      m, "Engine",
      "Reverse Monte Carlo refinement. Each step selects a frame and a group, "
      "proposes a move, and keeps it if the sampler accepts the new total "
      "chi^2 and no rigid constraint objects."};

  cls.def(py::init<AtomicStructure, BoundaryConditions, std::uint32_t>(),
          py::arg("structure"), py::arg("bc"), py::kw_only(),
          py::arg("frame_rng_seed") = std::uint32_t{1729},
          "Refine a copy of structure under bc. Bind constraints to "
          "engine.structure, not to the structure passed in.")
      .def_property_readonly(
          "structure", [](Engine &e) -> AtomicStructure & { return e.structure(); },
          "Frame 0, by reference: what the engine moves, and what constraints "
          "should borrow.")
      .def_property_readonly("boundary", &Engine::boundary)
      .def_property_readonly(
          "constraints",
          [](Engine &e) -> RMC::ConstraintCollection & { return e.constraints(); },
          "The engine's constraints, by reference; index them and call "
          ".concrete() to read computed curves.")
      .def_property_readonly("collector", &Engine::collector,
                             "Atoms removed by removal groups.")
      .def("add_frame", &Engine::add_frame, py::arg("structure"),
           "Refine a copy of another frame against the averaged profile.")
      .def("add_group", &Engine::add_group, py::arg("group"),
           py::keep_alive<1, 2>())
      .def(
          "add_constraint",
          [](Engine &e, Constraint c) {
            require_addable(e.constraints(), c);
            e.add_constraint(std::move(c));
          },
          py::arg("constraint"), py::keep_alive<1, 2>(),
          "Add a copy (read it back through engine.constraints). A second "
          "singular constraint of one kind raises ValueError.")
      .def("build_atomic_groups", &Engine::build_atomic_groups,
           py::arg("min_amp") = 0.0, py::arg("max_amp") = 0.2, py::kw_only(),
           py::arg("seed") = std::uint32_t{42},
           "One group per atom, each a TranslationGenerator(min_amp, max_amp). "
           "Replaces existing groups.")
      .def("build_langevin_groups", &Engine::build_langevin_groups,
           py::arg("step_size"), py::kw_only(), py::arg("seed") = std::uint32_t{42},
           "Per-atom Langevin (MALA) groups along -grad chi^2. Call after every "
           "constraint is added.")
      .def("build_leapfrog_groups", &Engine::build_leapfrog_groups,
           py::arg("step_size"), py::arg("n_steps") = 10, py::kw_only(),
           py::arg("seed") = std::uint32_t{42},
           "Per-atom HMC leapfrog groups. Call after every constraint is added.")
      .def("add_removal_group", &Engine::add_removal_group, py::arg("name"),
           py::arg("indices"),
           "A group whose move removes its atoms (committed if accepted).")
      .def(
          "set_checkpoint",
          [](Engine &e, std::filesystem::path path, std::uint64_t every) {
            e.set_checkpoint(std::move(path), require_positive(every, "every"));
          },
          py::arg("path"), py::kw_only(), py::arg("every") = 5000,
          "Save a checkpoint every `every` accepted moves.")
      .def("set_track_best", &Engine::set_track_best, py::arg("on") = true,
           "Keep the lowest-error configuration seen (best_structure).")
      .def_property_readonly("best_error", &Engine::best_error)
      .def_property_readonly(
          "best_structure",
          [](const Engine &e) { return AtomicStructure{e.best_structure()}; },
          "A copy of the lowest-error frame 0 seen (with set_track_best), else "
          "of the current frame 0.")
      .def("initialise", &Engine::initialise,
           "Prime the constraints; run() does this on first use.")
      .def(
          "run",
          [](Engine &e, std::uint64_t n_steps) {
            require_groups(e);
            const py::gil_scoped_release unlocked;
            e.run(n_steps);
          },
          py::arg("n_steps"), "Run n_steps steps (the GIL is released).")
      .def(
          "run_until",
          [](Engine &e, double target_chi2, std::uint64_t max_steps) {
            require_groups(e);
            const py::gil_scoped_release unlocked;
            e.run_until(target_chi2, max_steps);
          },
          py::arg("target_chi2"), py::arg("max_steps") = 0,
          "Run until the total chi^2 reaches target_chi2 (or max_steps; 0 is "
          "unbounded).")
      .def_property_readonly("total_error", &Engine::total_error)
      .def_property_readonly("steps_total", &Engine::steps_total)
      .def_property_readonly("steps_accepted", &Engine::steps_accepted)
      .def_property_readonly("n_groups", &Engine::n_groups)
      .def_property_readonly("stats", &Engine::stats);

  // A step callback sees the frame by reference, tied to the engine's Python
  // object: a structure the callback keeps holds the engine alive instead of
  // dangling. The owner handle is borrowed -- the callback only runs inside
  // run(), while the engine is alive -- so no reference cycle forms.
  cls.def(
         "set_step_callback",
         [](py::object self, py::function fn, std::uint64_t log_every) {
           self.cast<Engine &>().set_step_callback(
               [fn = std::move(fn), owner = self.ptr()](
                   std::uint64_t step, std::uint64_t accepted,
                   std::uint64_t tried, double chi2, const AtomicStructure &s) {
                 const py::gil_scoped_acquire gil;
                 fn(step, accepted, tried, chi2,
                    py::cast(s, py::return_value_policy::reference_internal,
                             py::handle{owner}));
               },
               require_positive(log_every, "log_every"));
         },
         py::arg("fn"), py::arg("log_every") = 1000,
         "Call fn(step, accepted, tried, chi2, structure) every log_every "
         "steps. Chi2CollectorCallback, PDBSnapshotCallback and "
         "HistogramCallback are ready-made ones.")
      .def(
          "clear_step_callback",
          [](Engine &e) { e.set_step_callback({}); }, "Remove the step callback.");

  // One overload per sampler and selector: generated from the variants, so a
  // new alternative in C++ is a new overload here.
  for_each_type<Alternatives<RMC::Sampler>>([&](auto id) {
    using S = typename decltype(id)::type;
    cls.def(
        "set_sampler",
        [](Engine &e, const S &sampler, std::uint32_t seed) {
          e.set_sampler(sampler, seed);
        },
        py::arg("sampler"), py::kw_only(), py::arg("seed") = std::uint32_t{0xACCE55},
        "Replace the acceptance policy; seed the uniforms Metropolis and "
        "annealing draw.");
  });
  for_each_type<Alternatives<RMC::GroupSelector>>([&](auto id) {
    using S = typename decltype(id)::type;
    cls.def(
           "set_selector",
           [](Engine &e, const S &selector) { e.set_selector(selector); },
           py::arg("selector"), "How each step picks a group.")
        .def(
            "set_frame_selector",
            [](Engine &e, const S &selector) { e.set_frame_selector(selector); },
            py::arg("selector"), "How each step picks a frame.");
  });

  cls.def("__repr__", [](Engine &e) {
    return std::format("Engine({} atoms, {} groups, {} constraints, {} steps)",
                       e.structure().size(), e.n_groups(), e.constraints().size(),
                       e.steps_total());
  });
}

// The ready-made step callbacks, each a Python callable an engine can take.
void bind_callbacks(py::module_ &m) {
  using RMC::callbacks::Chi2CollectorCallback;
  using RMC::callbacks::HistogramCallback;
  using RMC::callbacks::PDBSnapshotCallback;

  py::class_<Chi2CollectorCallback>(
      m, "Chi2CollectorCallback",
      "Records (step, chi2). finalize() -- or leaving a `with` block -- writes "
      "csv_path and prints an ASCII chart; without it that happens when the "
      "object is collected.")
      .def(py::init<std::filesystem::path>(),
           py::arg("csv_path") = std::filesystem::path{"chi2.csv"})
      .def("__call__", &Chi2CollectorCallback::operator(), py::arg("step"),
           py::arg("accepted"), py::arg("tried"), py::arg("chi2"),
           py::arg("structure"))
      .def("finalize", &Chi2CollectorCallback::finalize,
           "Write the CSV and print the chart; later calls do nothing.")
      .def_property_readonly("history", &Chi2CollectorCallback::history,
                             "(step, chi2) per call.")
      .def(
          "__enter__",
          [](Chi2CollectorCallback &c) -> Chi2CollectorCallback & { return c; },
          py::return_value_policy::reference_internal)
      .def("__exit__",
           [](Chi2CollectorCallback &c, const py::args &) { c.finalize(); });

  py::class_<PDBSnapshotCallback>(
      m, "PDBSnapshotCallback",
      "Writes dir/step_<step>.pdb on each call (dir is created).")
      .def(py::init([](std::filesystem::path dir) {
             return PDBSnapshotCallback{.dir = std::move(dir)};
           }),
           py::arg("dir"))
      .def_readwrite("dir", &PDBSnapshotCallback::dir)
      .def("__call__", &PDBSnapshotCallback::operator(), py::arg("step"),
           py::arg("accepted"), py::arg("tried"), py::arg("chi2"),
           py::arg("structure"));

  py::class_<HistogramCallback>(
      m, "HistogramCallback",
      "Writes dir/hist_<step>.csv (axis, computed, experimental) from "
      "getter() -> (computed, experimental) on each call.")
      .def(py::init([](RMC::vec_t axis, HistogramCallback::Getter getter,
                       std::filesystem::path dir) {
             return HistogramCallback{.axis = std::move(axis),
                                      .getter = std::move(getter),
                                      .dir = std::move(dir)};
           }),
           py::arg("axis"), py::arg("getter"), py::arg("dir"))
      .def_readwrite("axis", &HistogramCallback::axis)
      .def_readwrite("dir", &HistogramCallback::dir)
      .def("__call__", &HistogramCallback::operator(), py::arg("step"),
           py::arg("accepted"), py::arg("tried"), py::arg("chi2"),
           py::arg("structure"));
}

} // namespace

void bind_engine(py::module_ &m) {
  bind_callbacks(m);
  bind_engine_class(m);
}

} // namespace rmc::python
