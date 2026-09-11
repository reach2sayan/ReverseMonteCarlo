#include "casters.hpp"
#include "constraints.hpp"
#include "describe.hpp"
#include "trampolines.hpp"

#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/constraints/ClusterCorrelationConstraint.hpp>
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/constraints/CoordinationConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/GeometricConstraints.hpp>
#include <RMC/constraints/PairCorrelationConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <RMC/constraints/StructureFactorConstraint.hpp>

#include <boost/mp11/algorithm.hpp>
#include <boost/type_erasure/any_cast.hpp>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <functional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Constraints: the concrete kinds, the type-erased Constraint every engine
// holds, Python-defined constraints, and ConstraintCollection.
namespace rmc::python {

namespace {

using RMC::AtomicStructure;
using RMC::BoundaryConditions;
using RMC::Constraint;
using RMC::coords_t;

// Every concrete kind, in one list: it generates Constraint's converting
// constructors and implicit conversions, and drives Constraint.concrete().
using ConstraintTypes = boost::mp11::mp_list<
    RMC::BondConstraint, RMC::AngleConstraint, RMC::DihedralAngleConstraint,
    RMC::ImproperAngleConstraint, RMC::InterMolecularDistanceConstraint,
    RMC::IntraMolecularDistanceConstraint, RMC::CoordinationConstraint,
    RMC::PairDistributionConstraint, RMC::PairCorrelationConstraint,
    RMC::StructureFactorConstraint, RMC::ReducedStructureFactorConstraint,
    RMC::AngularDistributionConstraint, RMC::ClusterCorrelationConstraint>;

constexpr const char *kBorrows =
    "Borrows the structure's per-atom arrays: pass engine.structure. The "
    "structure is kept alive and must keep its atom count.";

// What every concrete constraint shares: identity, flags, cost, the boundary
// conditions it measures under, and compute_error for use outside an engine.
template <RMC::CConstraint T>
py::class_<T> bind_constraint(py::module_ &m, const char *name, const char *doc) {
  py::class_<T> cls{m, name, doc};
  cls.def_property_readonly("name", [](const T &c) { return std::string{c.name()}; })
      .def_property_readonly("is_rigid", [](const T &c) { return c.is_rigid(); })
      .def_property_readonly("is_singular",
                             [](const T &c) { return c.is_singular(); })
      .def_property_readonly("computation_cost",
                             [](const T &c) { return c.computation_cost(); })
      .def(
          "set_boundary_conditions",
          [](T &c, const BoundaryConditions &bc) { c.set_boundary_conditions(bc); },
          py::arg("bc"), py::keep_alive<1, 2>(),
          "Measure under bc (an engine sets its own when the constraint is "
          "added).")
      .def("initialise", [](T &c) { c.initialise(); },
           "Build lookup tables; an engine does this itself.")
      .def(
          "compute_error",
          [](T &c, const Eigen::Ref<const coords_t> &coords,
             const std::vector<std::size_t> &moved) {
            for (const std::size_t i : moved) {
              require_atom(i, static_cast<std::size_t>(coords.rows()));
            }
            const coords_t owned = coords;
            return c.compute_error(owned, moved);
          },
          py::arg("coords"), py::arg("moved"),
          "The error for coords, given the atoms moved since the last call "
          "(outside an engine; for testing and inspection).");
  return cls;
}

// set_elements(structure): borrows the element symbols.
template <class T> void def_set_elements(py::class_<T> &cls) {
  cls.def(
      "set_elements",
      [](T &c, const AtomicStructure &s) { c.set_elements(s.elements); },
      py::arg("structure"), py::keep_alive<1, 2>(), kBorrows);
}

// The curve-fitting kinds (pair function, Fourier, angular distribution):
// experimental data in, computed curve out, and whichever of the density,
// weight and shape knobs the kind has.
template <class T> void def_curve(py::class_<T> &cls) {
  cls.def(
      "set_experimental_data",
      [](T &c, const RMC::mat_t &data) { c.set_experimental_data(data); },
      py::arg("data"),
      "The target: column 0 is the grid (r, Q or angle), the rest values.");
  def_set_elements(cls);
  if constexpr (requires(T &c) { c.set_number_density(1.0); }) {
    cls.def(
        "set_number_density", [](T &c, double rho0) { c.set_number_density(rho0); },
        py::arg("rho0"), "Number density (atoms per cubic Angstrom).");
  }
  if constexpr (requires(T &c, const std::string &e) { c.set_weight(e, e, 1.0); }) {
    cls.def(
        "set_weight",
        [](T &c, const std::string &a, const std::string &b, double w) {
          c.set_weight(a, b, w);
        },
        py::arg("element1"), py::arg("element2"), py::arg("weight"),
        "Weight of the element1-element2 partial.");
  }
  if constexpr (requires(T &c) { c.set_shape_function(std::function<double(double)>{}); }) {
    cls.def(
           "set_shape_function",
           [](T &c, std::function<double(double)> fn) {
             c.set_shape_function(std::move(fn));
           },
           py::arg("fn"),
           "Envelope f(r) multiplied into the computed curve (evaluated once "
           "per bin, on first use).")
        .def(
            "set_spherical_shape",
            [](T &c, double d) { c.set_shape_function(RMC::spherical_shape_fn(d)); },
            py::arg("diameter"),
            "Spherical-particle envelope of the given diameter (Angstrom).")
        .def(
            "set_gaussian_shape",
            [](T &c, double s) { c.set_shape_function(RMC::gaussian_shape_fn(s)); },
            py::arg("sigma"), "Gaussian envelope exp(-r^2 / sigma^2).");
  }
  cls.def_property_readonly(
         "computed",
         [](const T &c) -> RMC::vec_t {
           if constexpr (requires { c.computed_G(); }) {
             return c.computed_G();
           } else {
             return c.computed();
           }
         },
         "The curve computed from the last evaluated coordinates.")
      .def_property_readonly(
          "experimental",
          [](const T &c) -> RMC::vec_t {
            if constexpr (requires { c.experimental_data(); }) {
              return c.experimental_data();
            } else {
              return c.experimental();
            }
          },
          "The target values, flattened.");
}

void bind_geometric(py::module_ &m) {
  const auto range = "Rigid: rejects any move that worsens a violation of "
                     "[lo, hi]; adds nothing to chi^2.";
  bind_constraint<RMC::BondConstraint>(m, "BondConstraint", range)
      .def(py::init<>())
      .def("add_bond", &RMC::BondConstraint::add_bond, py::arg("i"), py::arg("j"),
           py::arg("lo"), py::arg("hi"), "Keep |r_i - r_j| in [lo, hi] (Angstrom).");
  bind_constraint<RMC::AngleConstraint>(m, "AngleConstraint", range)
      .def(py::init<>())
      .def("add_angle", &RMC::AngleConstraint::add_angle, py::arg("i"),
           py::arg("j"), py::arg("k"), py::arg("lo"), py::arg("hi"),
           "Keep the i-j-k angle (vertex j) in [lo, hi] radians.");
  bind_constraint<RMC::DihedralAngleConstraint>(m, "DihedralAngleConstraint", range)
      .def(py::init<>())
      .def("add_dihedral", &RMC::DihedralAngleConstraint::add_dihedral,
           py::arg("i"), py::arg("j"), py::arg("k"), py::arg("l"), py::arg("lo"),
           py::arg("hi"), "Keep the i-j-k-l dihedral in [lo, hi] radians.");
  bind_constraint<RMC::ImproperAngleConstraint>(m, "ImproperAngleConstraint", range)
      .def(py::init<>())
      .def("add_improper", &RMC::ImproperAngleConstraint::add_improper,
           py::arg("i"), py::arg("j"), py::arg("k"), py::arg("l"), py::arg("lo"),
           py::arg("hi"), "Keep the i-j-k-l improper angle in [lo, hi] radians.");
}

template <RMC::DistanceScope S>
void bind_distance(py::module_ &m, const char *name, const char *doc) {
  using T = RMC::DistanceConstraint<S>;
  bind_constraint<T>(m, name, doc)
      .def(py::init<>())
      .def("set_minimum_distance", &T::set_minimum_distance, py::arg("element1"),
           py::arg("element2"), py::arg("d_min"),
           "Closest approach allowed between the two elements (Angstrom).")
      .def(
          "set_structure",
          [](T &c, const AtomicStructure &s) {
            c.set_structure(s.elements, s.molecule_ids);
          },
          py::arg("structure"), py::keep_alive<1, 2>(), kBorrows);
}

void bind_coordination(py::module_ &m) {
  auto cls = bind_constraint<RMC::CoordinationConstraint>(
      m, "CoordinationConstraint",
      "Rigid: keeps each shell's neighbour count in [min_cn, max_cn].");
  cls.def(py::init<>())
      .def("add_shell", &RMC::CoordinationConstraint::add_shell,
           py::arg("centre"), py::arg("neighbour_element"), py::arg("r_min"),
           py::arg("r_max"), py::arg("min_cn"), py::arg("max_cn"),
           "Atom centre must have [min_cn, max_cn] neighbour_element atoms "
           "within [r_min, r_max] Angstrom.");
  def_set_elements(cls);
}

template <class T>
void bind_pair(py::module_ &m, const char *name, const char *doc) {
  auto cls = bind_constraint<T>(m, name, doc);
  cls.def(py::init<>())
      .def("set_grid", &T::set_grid, py::arg("r_first"), py::arg("dr"),
           py::arg("n"), "The histogram grid: n bins of width dr from r_first.")
      .def(
          "set_molecule_ids",
          [](T &c, const AtomicStructure &s) { c.set_molecule_ids(s.molecule_ids); },
          py::arg("structure"), py::keep_alive<1, 2>(), kBorrows)
      .def("set_exclude_intra", &T::set_exclude_intra, py::arg("exclude"),
           "Skip pairs within one molecule (needs set_molecule_ids).");
  def_curve(cls);
}

template <class T>
void bind_fourier(py::module_ &m, const char *name, const char *doc) {
  auto cls = bind_constraint<T>(m, name, doc);
  cls.def(py::init<>());
  def_curve(cls);
}

void bind_angular(py::module_ &m) {
  using T = RMC::AngularDistributionConstraint;
  auto cls = bind_constraint<T>(
      m, "AngularDistributionConstraint",
      "Singular: fits the bond-angle distribution (per central/leg-pair "
      "partial) to a multi-column target.");
  cls.def(py::init<>())
      .def("set_cutoff", &T::set_cutoff, py::arg("max_distance"),
           "Bond cutoff (Angstrom).")
      .def("set_smoothing", &T::set_smoothing, py::arg("half_width"),
           "Boxcar smoothing half-width in bins (0 disables).")
      .def("set_scale_invariant", &T::set_scale_invariant, py::arg("on"),
           "Fit shape only, with the best overall scale.")
      .def("set_resync_interval", &T::set_resync_interval, py::arg("accepts"),
           "Full recompute every this many accepted moves (0 never).");
  def_curve(cls);
}

void bind_cluster(py::module_ &m) {
  using RMC::ClusterOrbit;
  using T = RMC::ClusterCorrelationConstraint;
  describe_struct<ClusterOrbit>(
      m, "ClusterOrbit",
      "Symmetry-equivalent instances of one cluster: flat_sites holds "
      "body * instance_count site indices, instance i at [i*body, (i+1)*body).")
      .def(
          "add_instance",
          [](ClusterOrbit &o, const std::vector<std::size_t> &sites) {
            o.add_instance(sites);
          },
          py::arg("sites"), "Append one instance; the first fixes body.")
      .def_property_readonly("instance_count", &ClusterOrbit::instance_count)
      .def_property_readonly(
          "instances",
          [](const ClusterOrbit &o) {
            return o.instances() | std::views::transform([](auto sites) {
                     return sites | std::ranges::to<std::vector>();
                   }) |
                   std::ranges::to<std::vector>();
          },
          "Each instance's sites.");

  bind_constraint<T>(
      m, "ClusterCorrelationConstraint",
      "Weighted chi^2 of cluster correlations against their targets (SQS "
      "search). Reads only the structure's elements.")
      .def(py::init<const AtomicStructure &, const T::SpeciesMap &,
                    std::vector<ClusterOrbit>>(),
           py::arg("structure"), py::arg("species_map"), py::arg("orbits"),
           py::keep_alive<1, 2>(),
           "Binary/linear basis: species_map gives each element's occupation "
           "value (typically +1/-1).")
      .def(py::init<const AtomicStructure &, std::unordered_map<std::string, int>,
                    RMC::CorrFuncTable, std::vector<ClusterOrbit>>(),
           py::arg("structure"), py::arg("occ_index"), py::arg("table"),
           py::arg("orbits"), py::keep_alive<1, 2>(),
           "Multicomponent basis: occ_index maps element to 0..m-1; table holds "
           "one (n_func, n_occ) block per site type.")
      .def("current_correlations", &T::current_correlations,
           "Each orbit's correlation for the current occupation.")
      .def_property_readonly("orbits", &T::orbits);
}

// Constraint.concrete(): the object inside, by reference, or the wrapped
// Python object for a Python-defined constraint.
py::object concrete(py::handle self) {
  auto &c = self.cast<Constraint &>();
  py::object out = py::none();
  for_each_type<ConstraintTypes>([&](auto id) {
    using T = typename decltype(id)::type;
    if (out.is_none()) {
      if (T *p = boost::type_erasure::any_cast<T *>(&c)) {
        out = py::cast(p, py::return_value_policy::reference_internal, self);
      }
    }
  });
  if (const auto *p = boost::type_erasure::any_cast<const PyConstraint *>(&c);
      out.is_none() && p != nullptr) {
    out = p->impl();
  }
  return out;
}

void bind_holder(py::module_ &m) {
  py::class_<Constraint> holder{
      m, "Constraint",
      "Any constraint, held by value. Every concrete constraint converts to "
      "it implicitly; wrap a Python object with Constraint(obj)."};
  for_each_type<ConstraintTypes>([&](auto id) {
    using T = typename decltype(id)::type;
    holder.def(py::init<const T &>(), py::arg("constraint"));
    py::implicitly_convertible<T, Constraint>();
  });
  // Last, so a concrete constraint never lands here.
  holder.def(py::init([](py::object impl) {
               return Constraint{PyConstraint{std::move(impl)}};
             }),
             py::arg("impl"),
             "Wrap a Python object with compute_error(coords, moved) -> float "
             "and optional rigid, cost, name, set_boundary_conditions(bc), "
             "initialise() (see rmc.ConstraintProtocol).");

  holder
      .def_property_readonly("name",
                             [](const Constraint &c) { return std::string{c.name()}; })
      .def_property_readonly("is_rigid", [](const Constraint &c) { return c.is_rigid(); })
      .def_property_readonly("is_singular",
                             [](const Constraint &c) { return c.is_singular(); })
      .def_property_readonly("computation_cost",
                             [](const Constraint &c) { return c.computation_cost(); })
      .def("standard_error", [](const Constraint &c) { return c.standard_error(); },
           "The error after the last move (0 for rigid).")
      .def("standard_error_before",
           [](const Constraint &c) { return c.standard_error_before(); },
           "The error before the last move (0 for rigid).")
      .def("should_reject", [](const Constraint &c) { return c.should_reject(); },
           "Whether the last move worsened this constraint.")
      .def(
          "compute_before_move",
          [](Constraint &c, const Eigen::Ref<const coords_t> &coords,
             const std::vector<std::size_t> &moved) {
            c.compute_before_move(coords_t{coords}, moved);
          },
          py::arg("coords"), py::arg("moved"))
      .def(
          "compute_after_move",
          [](Constraint &c, const Eigen::Ref<const coords_t> &coords,
             const std::vector<std::size_t> &moved) {
            c.compute_after_move(coords_t{coords}, moved);
          },
          py::arg("coords"), py::arg("moved"))
      .def("accept", [](Constraint &c) { c.accept(); })
      .def("reject", [](Constraint &c) { c.reject(); })
      .def("initialise", [](Constraint &c) { c.initialise(); })
      .def(
          "set_boundary_conditions",
          [](Constraint &c, const BoundaryConditions &bc) {
            c.set_boundary_conditions(bc);
          },
          py::arg("bc"), py::keep_alive<1, 2>())
      .def("concrete", &concrete,
           "The constraint inside, by reference (read its curves after a run), "
           "or the wrapped object for a Python-defined one.")
      .def("__repr__", [](const Constraint &c) {
        return std::format("Constraint({})", c.name());
      });
}

void bind_collection(py::module_ &m) {
  using RMC::ConstraintCollection;
  py::class_<ConstraintCollection>(
      m, "ConstraintCollection",
      "The constraints an engine evaluates, cheapest first. Iterate it or "
      "index it for each Constraint (by reference).")
      .def(py::init<>())
      .def(
          "add",
          [](ConstraintCollection &cc, Constraint c) {
            require_addable(cc, c);
            cc.add(std::move(c));
          },
          py::arg("constraint"), py::keep_alive<1, 2>(),
          "Add a copy. A second singular constraint of one kind raises "
          "ValueError.")
      .def("__len__", [](const ConstraintCollection &cc) { return cc.size(); })
      .def(
          "__iter__",
          [](const ConstraintCollection &cc) {
            return py::make_iterator(cc.begin(), cc.end());
          },
          py::keep_alive<0, 1>())
      .def(
          "__getitem__",
          [](const ConstraintCollection &cc, std::size_t i) -> const Constraint & {
            return cc[static_cast<std::ptrdiff_t>(require_atom(i, cc.size()))];
          },
          py::arg("i"), py::return_value_policy::reference_internal)
      .def_property_readonly("total_error", &ConstraintCollection::total_error,
                             "Sum of the soft constraints' errors after the last "
                             "move.")
      .def_property_readonly("total_error_before",
                             &ConstraintCollection::total_error_before)
      .def("error_breakdown", &ConstraintCollection::error_breakdown,
           "(name, error) per constraint.")
      .def("initialise_all", &ConstraintCollection::initialise_all);
}

} // namespace

void bind_constraints(py::module_ &m) {
  bind_geometric(m);
  bind_distance<RMC::DistanceScope::Inter>(
      m, "InterMolecularDistanceConstraint",
      "Rigid: minimum distances between atoms of different molecules.");
  bind_distance<RMC::DistanceScope::Intra>(
      m, "IntraMolecularDistanceConstraint",
      "Rigid: minimum distances between atoms of one molecule.");
  bind_coordination(m);
  bind_pair<RMC::PairDistributionConstraint>(
      m, "PairDistributionConstraint",
      "Singular: fits the total pair distribution G(r) to a two-column target.");
  bind_pair<RMC::PairCorrelationConstraint>(
      m, "PairCorrelationConstraint",
      "Singular: fits the pair correlation g(r) to a two-column target.");
  bind_fourier<RMC::StructureFactorConstraint>(
      m, "StructureFactorConstraint",
      "Singular: fits S(Q), Fourier-transformed from the computed G(r).");
  bind_fourier<RMC::ReducedStructureFactorConstraint>(
      m, "ReducedStructureFactorConstraint",
      "Singular: fits the reduced F(Q) = Q[S(Q) - 1].");
  bind_angular(m);
  bind_cluster(m);
  bind_holder(m);
  bind_collection(m);
}

} // namespace rmc::python
