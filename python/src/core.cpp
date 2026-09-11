#include "casters.hpp"
#include "describe.hpp"
#include "errors.hpp"

#include <RMC/RMCRunner.hpp>                      // MoveGenKind
#include <RMC/constraints/DistanceConstraint.hpp> // DistanceScope
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Parallel.hpp>
#include <RMC/core/RandomStructure.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/MoveGenerator.hpp> // SymmetryAxis
#include <RMC/io/LammpsReader.hpp>          // LammpsAtomStyle
#include <RMC/io/StructFormat.hpp>
#include <RMC/selectors/GroupSelector.hpp> // RecursiveMode

#include <seitz/data/element_data.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// The vocabulary every other binding is phrased in: enums, BoundaryConditions,
// AtomicStructure, AtomsCollector, random structures and the parallel knobs.
namespace rmc::python {

namespace {

using RMC::AtomicStructure;
using RMC::BoundaryConditions;
using RMC::coords_t;
using RMC::ivec_t;
using RMC::mat3_t;

void bind_enums(py::module_ &m) {
  describe_enum<RMC::SymmetryAxis>(
      m, "SymmetryAxis", "Cartesian axis a symmetry-restricted move acts along.");
  describe_enum<RMC::MoveGenKind>(
      m, "MoveGenKind",
      "Move proposer: the classic random walk, or gradient-driven Langevin "
      "(MALA) / Leapfrog (HMC) moves along -grad chi^2.");
  describe_enum<RMC::RecursiveMode>(
      m, "RecursiveMode",
      "When RecursiveGroupSelector repeats a group: after an accepted move "
      "(Refine) or a rejected one (Explore).");
  describe_enum<RMC::io::StructFormat>(m, "StructFormat",
                                       "A structure file format.");
  describe_enum<RMC::io::LammpsAtomStyle>(
      m, "LammpsAtomStyle", "The LAMMPS data-file atom_style of the Atoms section.");
  describe_enum<RMC::DistanceScope>(
      m, "DistanceScope",
      "Which atom pairs a distance constraint covers: between molecules "
      "(Inter) or within one (Intra).");
}

[[nodiscard]] std::string repr(const BoundaryConditions &bc) {
  if (!bc.periodic()) {
    return std::format("InfiniteBC(volume={:g})", bc.volume());
  }
  const mat3_t &box = bc.box();
  return std::format("PeriodicBC(a={:g}, b={:g}, c={:g}, volume={:g})",
                     box.col(0).norm(), box.col(1).norm(), box.col(2).norm(),
                     bc.volume());
}

void bind_boundary(py::module_ &m) {
  py::class_<BoundaryConditions>(
      m, "BoundaryConditions",
      "A periodic cell (box columns are the cell vectors) or open space with a "
      "given volume.")
      .def(py::init<const mat3_t &>(), py::arg("box"),
           "Periodic in all three directions; the columns of box are the cell "
           "vectors (Angstrom).")
      .def(py::init<double>(), py::arg("volume") = 1.0,
           "Open (non-periodic) space; volume sets the number density.")
      .def_property_readonly("periodic", &BoundaryConditions::periodic)
      .def_property_readonly("box", &BoundaryConditions::box,
                             "The cell matrix, read-only (identity when open).")
      .def_property_readonly("inv_box", &BoundaryConditions::inv_box)
      .def_property_readonly("volume", &BoundaryConditions::volume)
      .def("wrap", &BoundaryConditions::wrap, py::arg("r"),
           "Fold a Cartesian position into the cell (identity when open).")
      .def("min_image", &BoundaryConditions::min_image, py::arg("d"),
           "The minimum-image displacement (identity when open).")
      .def("__repr__", py::overload_cast<const BoundaryConditions &>(&repr));

  py::class_<RMC::PeriodicBC, BoundaryConditions>(
      m, "PeriodicBC", "A periodic cell; the columns of box are the cell vectors.")
      .def(py::init<const mat3_t &>(), py::arg("box"));
  py::class_<RMC::InfiniteBC, BoundaryConditions>(
      m, "InfiniteBC", "Open space; volume sets the number density.")
      .def(py::init<double>(), py::arg("volume") = 1.0);
}

// Per-atom Z from element symbols, 0 where the symbol is not an element: the
// rule the VASP reader and make_random_amorphous use.
[[nodiscard]] ivec_t atomic_numbers_of(const std::vector<std::string> &elements) {
  ivec_t z(static_cast<Eigen::Index>(elements.size()));
  std::ranges::transform(elements, z.begin(), [](const std::string &e) {
    return static_cast<int>(seitz::data::atomic_number(e).value_or(0));
  });
  return z;
}

// The metadata a reader leaves when a file names only elements: names and
// residues are the symbol, every atom is molecule 1.
[[nodiscard]] AtomicStructure
make_structure(const Eigen::Ref<const coords_t> &coordinates,
               std::vector<std::string> elements,
               std::optional<ivec_t> atomic_numbers,
               std::optional<std::vector<std::string>> names,
               std::optional<std::vector<std::string>> residues,
               std::optional<std::vector<std::size_t>> molecule_ids) {
  AtomicStructure s;
  s.coordinates = coordinates;
  const std::size_t n = s.size();
  require_per_atom(elements.size(), n, "elements");
  s.atomic_numbers = atomic_numbers ? std::move(*atomic_numbers)
                                    : atomic_numbers_of(elements);
  s.names = std::move(names).value_or(elements);
  s.residues = std::move(residues).value_or(elements);
  s.molecule_ids =
      std::move(molecule_ids).value_or(std::vector<std::size_t>(n, 1));
  s.elements = std::move(elements);
  for (const auto &[field, size] :
       std::initializer_list<std::pair<const char *, std::size_t>>{
           {"atomic_numbers", static_cast<std::size_t>(s.atomic_numbers.size())},
           {"names", s.names.size()},
           {"residues", s.residues.size()},
           {"molecule_ids", s.molecule_ids.size()}}) {
    require_per_atom(size, n, field);
  }
  return s;
}

[[nodiscard]] std::string repr(const AtomicStructure &s) {
  std::map<std::string, std::size_t> composition;
  for (const std::string &e : s.elements) {
    ++composition[e];
  }
  return std::format("AtomicStructure({} atoms, {})", s.size(), composition);
}

void bind_structure(py::module_ &m) {
  py::class_<AtomicStructure> cls{
      m, "AtomicStructure",
      "Atoms: coordinates plus per-atom metadata. Constraints and generators "
      "borrow its arrays, so the atom count is fixed once built -- every setter "
      "writes in place and refuses a different length."};

  cls.def(py::init(&make_structure), py::arg("coordinates"), py::arg("elements"),
          py::kw_only(), py::arg("atomic_numbers") = py::none(),
          py::arg("names") = py::none(), py::arg("residues") = py::none(),
          py::arg("molecule_ids") = py::none(),
          "coordinates is (N, 3) in Angstrom. atomic_numbers default from the "
          "element symbols; names and residues default to the symbols; every "
          "atom defaults to molecule 1.");

  // The two arrays are zero-copy views into the structure (writes land in the
  // C++ object and each view keeps it alive); assignment copies in place.
  cls.def_property(
      "coordinates",
      [](AtomicStructure &s) -> coords_t & { return s.coordinates; },
      [](AtomicStructure &s, const Eigen::Ref<const coords_t> &c) {
        require_per_atom(static_cast<std::size_t>(c.rows()), s.size(),
                         "coordinates");
        s.coordinates = c;
      },
      "(N, 3) float64 positions, a writable view into this structure.");
  cls.def_property(
      "atomic_numbers",
      [](AtomicStructure &s) -> ivec_t & { return s.atomic_numbers; },
      [](AtomicStructure &s, const Eigen::Ref<const ivec_t> &z) {
        require_per_atom(static_cast<std::size_t>(z.size()), s.size(),
                         "atomic_numbers");
        s.atomic_numbers = z;
      },
      "(N,) int32 species codes, a writable view into this structure.");

  // The per-atom lists come back as copies; assignment copies element-wise so
  // spans a constraint holds into the vector stay valid.
  const auto per_atom = [&cls](const char *name, auto AtomicStructure::*field,
                               const char *doc) {
    using Field =
        std::remove_cvref_t<decltype(std::declval<AtomicStructure &>().*field)>;
    cls.def_property(
        name, [field](const AtomicStructure &s) { return s.*field; },
        [field, name](AtomicStructure &s, const Field &values) {
          require_per_atom(values.size(), s.size(), name);
          std::ranges::copy(values, (s.*field).begin());
        },
        doc);
  };
  per_atom("elements", &AtomicStructure::elements, "Element symbols.");
  per_atom("names", &AtomicStructure::names, "Atom names (PDB name column).");
  per_atom("residues", &AtomicStructure::residues, "Residue names.");
  per_atom("molecule_ids", &AtomicStructure::molecule_ids,
           "Molecule id per atom; intra/inter-molecular constraints read it.");

  const auto copy = [](const AtomicStructure &s) { return AtomicStructure{s}; };
  cls.def("__len__", &AtomicStructure::size)
      .def(
          "distance",
          [](const AtomicStructure &s, std::size_t i, std::size_t j,
             const BoundaryConditions &bc) {
            return s.distance(require_atom(i, s.size()),
                              require_atom(j, s.size()), bc);
          },
          py::arg("i"), py::arg("j"), py::arg("bc"),
          "Distance between atoms i and j under bc (minimum image if periodic).")
      .def("copy", copy, "An independent deep copy.")
      .def("__copy__", copy)
      .def(
          "__deepcopy__",
          [copy](const AtomicStructure &s, const py::dict &) { return copy(s); },
          py::arg("memo"))
      .def("__repr__", py::overload_cast<const AtomicStructure &>(&repr));
}

void bind_collector(py::module_ &m) {
  using RMC::AtomsCollector;
  py::class_<AtomsCollector>(
      m, "AtomsCollector",
      "Atoms removed from the system by removal moves. Not constructible: an "
      "Engine owns one (Engine.collector).")
      .def("is_removed", &AtomsCollector::is_removed, py::arg("i"),
           "Removed by an accepted move.")
      .def("is_pending", &AtomsCollector::is_pending, py::arg("i"),
           "Staged for removal by the move in flight.")
      .def("absent", &AtomsCollector::absent, py::arg("i"),
           "Removed or staged: constraints skip absent atoms.")
      .def_property_readonly(
          "pending",
          [](const AtomsCollector &c) {
            return std::vector<RMC::index_t>(c.pending().begin(),
                                             c.pending().end());
          },
          "Indices staged by the move in flight.")
      .def_property_readonly("n_removed", &AtomsCollector::n_removed)
      .def("n_active", &AtomsCollector::n_active, py::arg("total"),
           "Atoms still present out of total.");
}

void bind_random(py::module_ &m) {
  using RMC::RandomStructure;
  py::class_<RandomStructure>(m, "RandomStructure",
                              "A random amorphous configuration and its cubic "
                              "periodic cell.")
      .def_readonly("structure", &RandomStructure::structure)
      .def_readonly("box", &RandomStructure::box, "The cell matrix, read-only.")
      .def("periodic_bc", &RandomStructure::periodic_bc,
           "PeriodicBC over box.")
      .def("__repr__", [](const RandomStructure &r) {
        return std::format("RandomStructure({}, box side {:g})",
                           repr(r.structure), r.box(0, 0));
      });

  m.def(
      "make_random_amorphous",
      [](const std::vector<std::string> &elements,
         const std::vector<std::size_t> &counts, double spacing,
         std::uint32_t seed) {
        return unwrap_nogil(error_types().random_structure, [&] {
          return RMC::make_random_amorphous(elements, counts, spacing, seed);
        });
      },
      py::arg("elements"), py::arg("counts"), py::kw_only(),
      py::arg("spacing") = 3.0, py::arg("seed") = std::uint32_t{42},
      "A random amorphous start (Zhu et al., Acta Mater. 262, 119456 (2024)): "
      "atoms on a body-centred grid of pitch spacing (Angstrom) in a cubic "
      "cell, exact composition assigned in random order. Raises "
      "RandomStructureError on mismatched or non-positive input.");
}

void bind_parallel(py::module_ &m) {
  m.def(
      "set_max_concurrency",
      [](int n) {
        if (n < 1) {
          throw py::value_error("set_max_concurrency needs n >= 1");
        }
        RMC::parallel::set_max_concurrency(n);
      },
      py::arg("n"),
      "Threads the parallel kernels may use (a no-op without TBB). Call "
      "between runs, never during one.");
  m.def("default_concurrency", &RMC::parallel::default_concurrency,
        "Threads the parallel kernels use by default.");
  m.def("allocated_cpus", &RMC::allocated_cpus,
        "CPUs allocated to this process: SLURM/PBS/LSF, else the hardware "
        "thread count.");
#if defined(RMC_USE_TBB)
  m.attr("has_tbb") = true;
#else
  m.attr("has_tbb") = false;
#endif
}

} // namespace

void bind_core(py::module_ &m) {
  bind_enums(m);
  bind_boundary(m);
  bind_structure(m);
  bind_collector(m);
  bind_random(m);
  bind_parallel(m);
}

} // namespace rmc::python
