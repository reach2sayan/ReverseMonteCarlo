#include "describe.hpp"
#include "errors.hpp"

#include <RMC/analysis/AngularDistribution.hpp>
#include <RMC/analysis/Composition.hpp>
#include <RMC/analysis/RadialDistribution.hpp>
#include <RMC/core/Structure.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <filesystem>
#include <string>
#include <vector>

// Structure analysis: g(r) and the bond-angle distribution, total and
// per-partial. Every failure raises AnalysisError.
namespace rmc::python {

namespace {

namespace an = RMC::analysis;
using RMC::AtomicStructure;
using RMC::BoundaryConditions;

constexpr auto kAnalysis = &ErrorTypes::analysis;

// GrResult and AdfResult: a grid, the total curve, labelled partials, the
// composition and one scalar. Arrays are read-only views; as_dict() is the
// flat form a DataFrame takes.
template <class T, class Grid, class Scalar>
py::class_<T> bind_curves(py::module_ &m, const char *name, const char *doc,
                          const char *grid, Grid T::*grid_field,
                          const char *scalar, Scalar T::*scalar_field) {
  py::class_<T> cls{m, name, doc};
  cls.def_readonly(grid, grid_field)
      .def_readonly("total", &T::total)
      .def_readonly("partials", &T::partials)
      .def_readonly(scalar, scalar_field)
      .def_property_readonly(
          "species",
          [](const T &r) {
            return std::vector<an::SpeciesCount>(r.species.begin(),
                                                 r.species.end());
          },
          "(symbol, count) per species, sorted by symbol.")
      .def(
          "as_dict",
          [grid, grid_field](const T &r) {
            py::dict out;
            out[grid] = py::cast(r.*grid_field);
            out["total"] = py::cast(r.total);
            for (const an::LabeledCurve &c : r.partials) {
              out[py::str(c.label)] = py::cast(c.values);
            }
            return out;
          },
          "{grid, 'total', partial label...} -> array, one column each.");
  return cls;
}

void bind_types(py::module_ &m) {
  describe_struct<an::GrParams>(
      m, "GrParams",
      "g(r) histogram: n_bins bins over [r_min, r_max] Angstrom; "
      "exclude_intra skips pairs within one molecule.");
  describe_struct<an::AdfParams>(
      m, "AdfParams",
      "Bond-angle histogram: bonds shorter than max_dis Angstrom, n_bins over "
      "0-180 degrees, boxcar smoothing half-width smooth_range.");
  describe_struct<an::SpeciesCount>(m, "SpeciesCount", "A species and its count.");
  describe_struct<an::LabeledCurve>(m, "LabeledCurve",
                                    "A partial curve and its label (e.g. 'Cu-Zr').");
  bind_curves<an::GrResult>(m, "GrResult",
                            "Radial distribution: total g(r) and partials.",
                            "r", &an::GrResult::r, "density",
                            &an::GrResult::density);
  bind_curves<an::AdfResult>(
      m, "AdfResult",
      "Bond-angle distribution: total and per central/leg-pair partials.",
      "theta", &an::AdfResult::theta, "max_dis", &an::AdfResult::max_dis);
}

void bind_functions(py::module_ &m) {
  using PathGr = RMC::Result<an::GrResult> (*)(
      const std::filesystem::path &, const BoundaryConditions &,
      const an::GrParams &, const std::vector<std::string> &);
  using PathAdf = RMC::Result<an::AdfResult> (*)(
      const std::filesystem::path &, const BoundaryConditions &,
      const an::AdfParams &, const std::vector<std::string> &);
  const std::vector<std::string> no_legend;

  m.def(
      "compute_gr",
      [](const AtomicStructure &s, const BoundaryConditions &bc,
         const an::GrParams &params) {
        return unwrap_nogil(error_types().analysis, [&] {
          return an::compute_gr(s.coordinates, bc, s.elements, params);
        });
      },
      py::arg("structure"), py::arg("bc"), py::kw_only(),
      py::arg("params") = an::GrParams{}, "g(r) of structure under bc.");
  m.def("compute_gr", checked<static_cast<PathGr>(&an::compute_gr)>(kAnalysis),
        py::arg("path"), py::arg("bc"), py::kw_only(),
        py::arg("params") = an::GrParams{}, py::arg("type_to_element") = no_legend,
        "g(r) of the structure file at path (bc applies when the file has no "
        "cell).");
  m.def("write_gr", checked<&an::write_gr>(kAnalysis), py::arg("result"),
        py::arg("path"), "Write g(r) as columns: r, total, partials.");

  m.def(
      "compute_adf",
      [](const AtomicStructure &s, const BoundaryConditions &bc,
         const an::AdfParams &params) {
        return unwrap_nogil(error_types().analysis, [&] {
          return an::compute_adf(s.coordinates, bc, s.elements, params);
        });
      },
      py::arg("structure"), py::arg("bc"), py::kw_only(),
      py::arg("params") = an::AdfParams{},
      "Bond-angle distribution of structure under bc.");
  m.def("compute_adf", checked<static_cast<PathAdf>(&an::compute_adf)>(kAnalysis),
        py::arg("path"), py::arg("bc"), py::kw_only(),
        py::arg("params") = an::AdfParams{}, py::arg("type_to_element") = no_legend,
        "Bond-angle distribution of the structure file at path.");
  m.def("write_adf", checked<&an::write_adf>(kAnalysis), py::arg("result"),
        py::arg("path"), "Write the ADF as columns: theta, total, partials.");
}

} // namespace

void bind_analysis(py::module_ &m) {
  bind_types(m);
  bind_functions(m);
}

} // namespace rmc::python
