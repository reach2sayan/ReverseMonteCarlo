#include "describe.hpp"
#include "errors.hpp"

#include <RMC/Ensemble.hpp>
#include <RMC/RMCRunner.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <ranges>
#include <string>
#include <vector>

// Ensembles of replicas, and the RMC_run pipeline as functions over RMCConfig.
namespace rmc::python {

namespace {

using RMC::Engine;
using RMC::ExperimentalData;
using RMC::RMCConfig;

constexpr auto kConfig = &ErrorTypes::config;

// The engines make_engine(i) returns, held by their Python objects. Replicas
// run in place, so no engine a caller can still see is ever moved from, and
// the winner comes back as the very object make_engine returned.
struct Replicas {
  std::vector<py::object> owners;
  std::vector<Engine *> engines;
};

[[nodiscard]] Replicas make_replicas(const py::function &make_engine,
                                     std::size_t n_replicas) {
  if (n_replicas == 0) {
    throw py::value_error("an ensemble needs n_replicas >= 1");
  }
  Replicas r;
  for (const std::size_t i : std::views::iota(std::size_t{0}, n_replicas)) {
    py::object owner = make_engine(i);
    if (!py::isinstance<Engine>(owner)) {
      throw py::type_error(std::format("make_engine must return an Engine, not {}",
                                       py::type::of(owner).attr("__name__").cast<std::string>()));
    }
    auto &engine = owner.cast<Engine &>();
    if (engine.n_groups() == 0) {
      throw py::value_error(std::format("replica {} has no groups", i));
    }
    if (std::ranges::contains(r.engines, &engine)) {
      throw py::value_error("make_engine returned the same engine twice");
    }
    r.engines.push_back(&engine);
    r.owners.push_back(std::move(owner));
  }
  return r;
}

py::object run_ensemble(const py::function &make_engine, std::size_t n_replicas,
                        std::uint64_t n_steps, std::size_t tbb_threads,
                        const py::object &prepare) {
  Replicas r = make_replicas(make_engine, n_replicas);
  if (!prepare.is_none()) {
    std::ranges::for_each(r.owners, [&](const py::object &e) { prepare(e); });
  }
  {
    const py::gil_scoped_release unlocked;
    RMC::detail::run_replicas(n_replicas, tbb_threads,
                              [&](std::size_t i) { r.engines[i]->run(n_steps); });
  }
  const auto best = std::ranges::min_element(r.engines, {}, &Engine::best_error);
  return r.owners[static_cast<std::size_t>(best - r.engines.begin())];
}

py::object run_ensemble_cooperative(const py::function &make_engine,
                                    std::size_t n_replicas, double target_chi2,
                                    std::uint64_t sync_every,
                                    std::uint64_t max_steps,
                                    std::size_t tbb_threads) {
  if (sync_every == 0) {
    throw py::value_error("sync_every must be >= 1");
  }
  Replicas r = make_replicas(make_engine, n_replicas);
  std::size_t best = 0;
  {
    const py::gil_scoped_release unlocked;
    best = RMC::run_cooperative(r.engines, target_chi2, sync_every, max_steps,
                                tbb_threads);
  }
  return r.owners[best];
}

void bind_ensembles(py::module_ &m) {
  m.def("run_ensemble", &run_ensemble, py::arg("make_engine"),
        py::arg("n_replicas"), py::arg("n_steps"), py::kw_only(),
        py::arg("tbb_threads") = std::size_t{0}, py::arg("prepare") = py::none(),
        "Run make_engine(i) for i in range(n_replicas), then each engine for "
        "n_steps on its own thread (GIL released), and return the one with "
        "the lowest best_error. prepare(engine) runs on each first -- where "
        "gradient groups are built. tbb_threads: TBB workers per replica "
        "(0 = the allocated CPUs split evenly).");
  m.def("run_ensemble_cooperative", &run_ensemble_cooperative,
        py::arg("make_engine"), py::arg("n_replicas"), py::arg("target_chi2"),
        py::kw_only(), py::arg("sync_every") = std::uint64_t{1000},
        py::arg("max_steps") = std::uint64_t{0},
        py::arg("tbb_threads") = std::size_t{0},
        "Replicas that share progress: every sync_every steps the best "
        "structure is copied into the others. Stops at target_chi2 (or "
        "max_steps; 0 is unbounded) and returns the engine with the lowest "
        "current chi^2.");
}

void bind_config(py::module_ &m) {
  describe_struct<RMCConfig>(
      m, "RMCConfig",
      "The RMC_run knobs as plain fields; its defaults are the CLI's. Exactly "
      "one of pdb_path, lammps_path, vasp_path names the structure; an empty "
      "path means not supplied. rmc.RMCConfig is the validated model over it.");
  describe_struct<ExperimentalData>(
      m, "ExperimentalData",
      "The experimental targets RMCConfig names: pdf and sq as (N, 2) arrays, "
      "adf as an (N, k) array, each None when not supplied.");

  m.def("load_structure", checked<&RMC::load_structure>(kConfig),
        py::arg("config"),
        "Read the structure config names, with box_override applied.");
  m.def("load_experimental_data", checked<&RMC::load_experimental_data>(kConfig),
        py::arg("config"), "Read the targets config names.");
  m.def("attach_constraints", &RMC::attach_constraints, py::arg("engine"),
        py::arg("data"), py::arg("config"),
        "Add a constraint per target in data, configured from config.");
  m.def("build_engine",
        static_cast<Engine (*)(const RMC::LoadedStructure &,
                               const ExperimentalData &, const RMCConfig &)>(
            &RMC::build_engine),
        py::arg("loaded"), py::arg("data"), py::arg("config"),
        "An engine over a copy of loaded: per-atom translation groups, the "
        "smart selector if config.use_smart, a constraint per target. Vary "
        "config.seed to build replicas from one read.");
  m.def("build_engine",
        checked<static_cast<RMC::Result<Engine> (*)(const RMCConfig &)>(
            &RMC::build_engine)>(kConfig),
        py::arg("config"), "Load everything config names, then build_engine.");
  m.def("apply_move_generator", &RMC::apply_move_generator, py::arg("engine"),
        py::arg("config"),
        "Rebuild the groups with config.move_gen's gradient proposer (Random is "
        "a no-op). Call after build_engine, before run.");
  m.def("periodic_box_or_zero", &RMC::periodic_box_or_zero, py::arg("bc"),
        "bc's cell matrix, or zeros when it is open.");
  m.def(
      "write_structure_by_ext",
      [](const RMC::AtomicStructure &s, const RMC::mat3_t &box,
         const std::filesystem::path &path) {
        unwrap_nogil(error_types().io, [&] {
          return RMC::write_structure_by_ext(s, box, path.string());
        });
      },
      py::arg("structure"), py::arg("box"), py::arg("path"),
      "Write structure in the format path's extension names (.vasp/.poscar, "
      ".lammps/.lmp/.data, else PDB); box is the periodic cell or zeros.");
}

} // namespace

void bind_runner(py::module_ &m) {
  bind_ensembles(m);
  bind_config(m);
}

} // namespace rmc::python
