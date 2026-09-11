#include "casters.hpp"
#include "describe.hpp"
#include "trampolines.hpp"

#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/generators/Agitations.hpp>
#include <RMC/generators/Combined.hpp>
#include <RMC/generators/LangevinRotationGenerator.hpp>
#include <RMC/generators/LangevinTranslationGenerator.hpp>
#include <RMC/generators/LeapfrogTranslationGenerator.hpp>
#include <RMC/generators/Path.hpp>
#include <RMC/generators/Removes.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/SpeciesSwap.hpp>
#include <RMC/generators/Swaps.hpp>
#include <RMC/generators/Translations.hpp>

#include <boost/mp11/algorithm.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Move generators: the concrete kinds, the type-erased MoveGenerator a Group
// holds, Python-defined generators, and Group itself.
namespace rmc::python {

namespace {

using RMC::coords_t;
using RMC::MoveGenerator;
using RMC::SymmetryAxis;
using RMC::vec3_t;

// Every concrete kind, in one list: it generates MoveGenerator's converting
// constructors and implicit conversions.
using GeneratorTypes = boost::mp11::mp_list<
    RMC::TranslationGenerator, RMC::TranslationAlongAxisGenerator,
    RMC::TranslationAlongSymmetryAxisGenerator,
    RMC::TranslationTowardsCentreGenerator, RMC::TranslationTowardsAxisGenerator,
    RMC::TranslationTowardsSymmetryAxisGenerator, RMC::RotationGenerator,
    RMC::RotationAboutAxisGenerator, RMC::RotationAboutSymmetryAxisGenerator,
    RMC::OrientationGenerator, RMC::DistanceAgitationGenerator,
    RMC::AngleAgitationGenerator, RMC::SwapGenerator, RMC::SwapCentersGenerator,
    RMC::TranslationAlongAxisPath, RMC::RotationAboutAxisPath,
    RMC::MoveGeneratorCollector, RMC::RemoveGenerator,
    RMC::SpeciesSwapGenerator, RMC::LangevinTranslationGenerator,
    RMC::LangevinRotationGenerator, RMC::LeapfrogTranslationGenerator>;

constexpr std::uint32_t kSeed = 42;

// One move on coords, in place. generate() takes the engine's own coords_t,
// which a numpy array is not, so it runs on a copy that is written back.
template <class G>
void generate_into(G &g, Eigen::Ref<coords_t> coords,
                   const std::vector<std::size_t> &indices) {
  for (const std::size_t i : indices) {
    require_atom(i, static_cast<std::size_t>(coords.rows()));
  }
  coords_t work = coords;
  g.generate(work, indices);
  coords = work;
}

// What every concrete generator shares: generate() for use outside an engine,
// and its amplitude range when it has one. Bases keeps the Python class
// hierarchy (a symmetry-axis generator is an axis generator).
template <class T, class... Bases>
py::class_<T, Bases...> bind_generator(py::module_ &m, const char *name,
                                       const char *doc) {
  py::class_<T, Bases...> cls{m, name, doc};
  cls.def("generate", &generate_into<T>, py::arg("coords"), py::arg("indices"),
          "Apply one move to the (N, 3) float64 array coords, in place "
          "(outside an engine; for testing).");
  if constexpr (sizeof...(Bases) == 0) {
    if constexpr (requires { &T::amp; }) {
      cls.def_readwrite("amp", &T::amp, "The move's amplitude range.");
    }
    if constexpr (requires { &T::angle; }) {
      cls.def_readwrite("angle", &T::angle, "The rotation's angle range (rad).");
    }
  }
  return cls;
}

void bind_translations(py::module_ &m) {
  bind_generator<RMC::TranslationGenerator>(
      m, "TranslationGenerator",
      "Translate the group by a distance in [lo, hi] (Angstrom) along a random "
      "direction.")
      .def(py::init<double, double, std::uint32_t>(), py::arg("lo"), py::arg("hi"),
           py::kw_only(), py::arg("seed") = kSeed);
  bind_generator<RMC::TranslationAlongAxisGenerator>(
      m, "TranslationAlongAxisGenerator",
      "Translate the group along axis by +/- a distance in [lo, hi].")
      .def(py::init<const vec3_t &, double, double, std::uint32_t>(),
           py::arg("axis"), py::arg("lo"), py::arg("hi"), py::kw_only(),
           py::arg("seed") = kSeed);
  bind_generator<RMC::TranslationAlongSymmetryAxisGenerator,
                 RMC::TranslationAlongAxisGenerator>(
      m, "TranslationAlongSymmetryAxisGenerator",
      "TranslationAlongAxisGenerator along a Cartesian axis.")
      .def(py::init<SymmetryAxis, double, double, std::uint32_t>(),
           py::arg("axis"), py::arg("lo"), py::arg("hi"), py::kw_only(),
           py::arg("seed") = kSeed);
  bind_generator<RMC::TranslationTowardsCentreGenerator>(
      m, "TranslationTowardsCentreGenerator",
      "Step the group's centroid towards centre by a distance in [lo, hi].")
      .def(py::init<const vec3_t &, double, double, std::uint32_t>(),
           py::arg("centre"), py::arg("lo"), py::arg("hi"), py::kw_only(),
           py::arg("seed") = kSeed);
  bind_generator<RMC::TranslationTowardsAxisGenerator>(
      m, "TranslationTowardsAxisGenerator",
      "Step the group's centroid towards the line through point along "
      "direction.")
      .def(py::init<const vec3_t &, const vec3_t &, double, double, std::uint32_t>(),
           py::arg("point"), py::arg("direction"), py::arg("lo"), py::arg("hi"),
           py::kw_only(), py::arg("seed") = kSeed);
  bind_generator<RMC::TranslationTowardsSymmetryAxisGenerator,
                 RMC::TranslationTowardsAxisGenerator>(
      m, "TranslationTowardsSymmetryAxisGenerator",
      "TranslationTowardsAxisGenerator towards a Cartesian axis through the "
      "origin.")
      .def(py::init<SymmetryAxis, double, double, std::uint32_t>(),
           py::arg("axis"), py::arg("lo"), py::arg("hi"), py::kw_only(),
           py::arg("seed") = kSeed);
}

void bind_rotations(py::module_ &m) {
  bind_generator<RMC::RotationGenerator>(
      m, "RotationGenerator",
      "Rotate the group about its centroid by +/- an angle in [lo, hi] (rad) "
      "about a random axis.")
      .def(py::init<double, double, std::uint32_t>(), py::arg("lo"), py::arg("hi"),
           py::kw_only(), py::arg("seed") = kSeed);
  bind_generator<RMC::RotationAboutAxisGenerator>(
      m, "RotationAboutAxisGenerator",
      "Rotate the group about its centroid around axis.")
      .def(py::init<const vec3_t &, double, double, std::uint32_t>(),
           py::arg("axis"), py::arg("lo"), py::arg("hi"), py::kw_only(),
           py::arg("seed") = kSeed);
  bind_generator<RMC::RotationAboutSymmetryAxisGenerator,
                 RMC::RotationAboutAxisGenerator>(
      m, "RotationAboutSymmetryAxisGenerator",
      "RotationAboutAxisGenerator around a Cartesian axis.")
      .def(py::init<SymmetryAxis, double, double, std::uint32_t>(),
           py::arg("axis"), py::arg("lo"), py::arg("hi"), py::kw_only(),
           py::arg("seed") = kSeed);
  bind_generator<RMC::OrientationGenerator>(
      m, "OrientationGenerator",
      "Turn the group's first-to-last axis towards target, within max_offset "
      "(rad).")
      .def(py::init<const vec3_t &, double, std::uint32_t>(), py::arg("target"),
           py::arg("max_offset"), py::kw_only(), py::arg("seed") = kSeed);
}

void bind_agitations_and_swaps(py::module_ &m) {
  bind_generator<RMC::DistanceAgitationGenerator>(
      m, "DistanceAgitationGenerator",
      "Stretch or shrink the i-j bond by +/- [lo, hi], keeping its midpoint.")
      .def(py::init<std::size_t, std::size_t, double, double, std::uint32_t>(),
           py::arg("i"), py::arg("j"), py::arg("lo"), py::arg("hi"), py::kw_only(),
           py::arg("seed") = kSeed);
  bind_generator<RMC::AngleAgitationGenerator>(
      m, "AngleAgitationGenerator",
      "Open or close the i-j-k angle by +/- [lo, hi] rad, keeping both bond "
      "lengths.")
      .def(py::init<std::size_t, std::size_t, std::size_t, double, double,
                    std::uint32_t>(),
           py::arg("i"), py::arg("j"), py::arg("k"), py::arg("lo"), py::arg("hi"),
           py::kw_only(), py::arg("seed") = kSeed);
  bind_generator<RMC::SwapGenerator>(
      m, "SwapGenerator",
      "Swap the group's positions with one candidate group of the same size.")
      .def(py::init<std::vector<std::vector<std::size_t>>, std::uint32_t>(),
           py::arg("candidates"), py::kw_only(), py::arg("seed") = kSeed);
  bind_generator<RMC::SwapCentersGenerator>(
      m, "SwapCentersGenerator",
      "Move the group's centroid onto a random candidate group's centroid.")
      .def(py::init<std::vector<std::vector<std::size_t>>, std::uint32_t>(),
           py::arg("candidates"), py::kw_only(), py::arg("seed") = kSeed);
}

void bind_paths(py::module_ &m) {
  bind_generator<RMC::TranslationAlongAxisPath>(
      m, "TranslationAlongAxisPath",
      "Translate along axis by each displacement in turn, cycling.")
      .def(py::init<const vec3_t &, std::vector<double>>(), py::arg("axis"),
           py::arg("displacements"));
  bind_generator<RMC::RotationAboutAxisPath>(
      m, "RotationAboutAxisPath",
      "Rotate about axis (through the centroid) by each angle in turn, "
      "cycling.")
      .def(py::init<const vec3_t &, std::vector<double>>(), py::arg("axis"),
           py::arg("angles"));
}

// Generators that hold a pointer into something else: kept alive by the
// generator's Python object; the target must stay where it is.
void bind_borrowing(py::module_ &m) {
  bind_generator<RMC::RemoveGenerator>(
      m, "RemoveGenerator",
      "Remove the group's atoms (staged; committed if the move is accepted). "
      "Pass engine.collector; Engine.add_removal_group builds one for you.")
      .def(py::init<RMC::AtomsCollector *>(), py::arg("collector"),
           py::keep_alive<1, 2>());
  bind_generator<RMC::SpeciesSwapGenerator>(
      m, "SpeciesSwapGenerator",
      "Swap the species of the group's first site with a differing site on "
      "its sublattice. Pass engine.structure.")
      .def(py::init<RMC::AtomicStructure &, std::vector<std::vector<std::size_t>>,
                    std::uint32_t>(),
           py::arg("structure"), py::arg("sublattices"), py::kw_only(),
           py::arg("seed") = kSeed, py::keep_alive<1, 2>());

  const char *gradient =
      "Build after every constraint is added, from engine.constraints, and "
      "only once the engine is where it will run (Engine.build_*_groups or "
      "run_ensemble's prepare do this for you).";
  bind_generator<RMC::LangevinTranslationGenerator>(
      m, "LangevinTranslationGenerator",
      std::format("Langevin (MALA) translation along -grad chi^2 (finite "
                  "differences). {}",
                  gradient)
          .c_str())
      .def(py::init<double, RMC::ConstraintCollection &, std::uint32_t>(),
           py::arg("step_size"), py::arg("constraints"), py::kw_only(),
           py::arg("seed") = kSeed, py::keep_alive<1, 3>());
  bind_generator<RMC::LangevinRotationGenerator>(
      m, "LangevinRotationGenerator",
      std::format("Langevin rotation along -dchi^2/dtheta. {}", gradient).c_str())
      .def(py::init<double, RMC::ConstraintCollection &, std::uint32_t>(),
           py::arg("step_size"), py::arg("constraints"), py::kw_only(),
           py::arg("seed") = kSeed, py::keep_alive<1, 3>());
  bind_generator<RMC::LeapfrogTranslationGenerator>(
      m, "LeapfrogTranslationGenerator",
      std::format("Hamiltonian Monte Carlo (leapfrog) translation; runs its own "
                  "accept/reject. {}",
                  gradient)
          .c_str())
      .def(py::init<int, double, RMC::ConstraintCollection &, std::uint32_t>(),
           py::arg("n_steps"), py::arg("step_size"), py::arg("constraints"),
           py::kw_only(), py::arg("seed") = kSeed, py::keep_alive<1, 4>());
}

void bind_holder(py::module_ &m) {
  py::class_<MoveGenerator> holder{
      m, "MoveGenerator",
      "Any move generator, held by value. Every concrete generator converts to "
      "it implicitly; wrap a Python object with MoveGenerator(obj)."};

  // MoveGeneratorCollector takes MoveGenerators, so it is bound once the
  // holder exists and after it the converting constructors.
  bind_generator<RMC::MoveGeneratorCollector>(
      m, "MoveGeneratorCollector",
      "Pick one of its generators per move, with probability proportional to "
      "its weight.")
      .def(py::init<std::uint32_t>(), py::kw_only(), py::arg("seed") = kSeed)
      .def("add", &RMC::MoveGeneratorCollector::add, py::arg("generator"),
           py::arg("weight") = 1.0, py::keep_alive<1, 2>(),
           "Add a generator (non-positive weights count as 1).");

  for_each_type<GeneratorTypes>([&](auto id) {
    using T = typename decltype(id)::type;
    holder.def(py::init<const T &>(), py::arg("generator"));
    py::implicitly_convertible<T, MoveGenerator>();
  });
  // Last, so a concrete generator never lands here.
  holder.def(py::init([](py::object impl) {
               return MoveGenerator{PyMoveGenerator{std::move(impl)}};
             }),
             py::arg("impl"),
             "Wrap a Python object with generate(coords, indices) -> None and "
             "optional modifies_species, rejection_override() (see "
             "rmc.MoveGeneratorProtocol).");

  holder.def("generate", &generate_into<MoveGenerator>, py::arg("coords"),
             py::arg("indices"), "Apply one move to coords, in place.")
      .def_property_readonly(
          "modifies_species",
          [](const MoveGenerator &g) { return g.modifies_species(); })
      .def(
          "rejection_override",
          [](const MoveGenerator &g) { return g.rejection_override(); },
          "The generator's own accept/reject verdict, or None.");
}

void bind_group(py::module_ &m) {
  using RMC::Group;
  py::class_<Group>(m, "Group",
                    "Atoms moved together by one generator. refine=False "
                    "freezes the group.")
      .def(py::init([](std::string name, std::vector<std::size_t> indices,
                       std::optional<MoveGenerator> generator, bool refine) {
             return Group{.name = std::move(name),
                          .indices = std::move(indices),
                          .generator = std::move(generator),
                          .refine = refine};
           }),
           py::arg("name"), py::arg("indices"), py::arg("generator") = py::none(),
           py::kw_only(), py::arg("refine") = true, py::keep_alive<1, 4>())
      .def_readwrite("name", &Group::name)
      .def_readwrite("indices", &Group::indices)
      .def_readwrite("generator", &Group::generator)
      .def_readwrite("refine", &Group::refine)
      .def("__len__", &Group::size)
      .def("__repr__", [](const Group &g) {
        return std::format("Group('{}', {} atoms{}{})", g.name, g.size(),
                           g.generator ? "" : ", no generator",
                           g.refine ? "" : ", frozen");
      });
}

} // namespace

void bind_generators(py::module_ &m) {
  describe_struct<RMC::Amplitude>(
      m, "Amplitude", "A move's magnitude range [lo, hi]; lo == hi is fixed.");
  bind_translations(m);
  bind_rotations(m);
  bind_agitations_and_swaps(m);
  bind_paths(m);
  bind_borrowing(m);
  bind_holder(m);
  bind_group(m);
}

} // namespace rmc::python
