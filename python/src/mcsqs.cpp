#include "describe.hpp"
#include "errors.hpp"

#include <AtatFormats.hpp>
#include <SeitzClusters.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <concepts>
#include <cstdint>
#include <filesystem>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// rmc._core.mcsqs: the SQS pipeline -- ATAT lattices in, a supercell and its
// cluster orbits (enumerated on seitz) out. rmc.mcsqs adds the engine setup.
namespace rmc::python {

namespace {

namespace atat = RMC::atat;

// The pipeline reports failure by throwing std::runtime_error; raise it as a
// McsqsError (pybind11 would flatten it to RuntimeError). A GIL released
// inside `call` is re-acquired before the handler runs.
template <std::invocable F> auto mcsqs_call(F &&call) -> decltype(call()) {
  try {
    return call();
  } catch (const std::runtime_error &e) {
    detail::raise(error_types().mcsqs, e.what());
  }
}

void bind_types(py::module_ &sub) {
  describe_struct<atat::LatticeSite>(
      sub, "LatticeSite",
      "A lattice site: fractional position and the (species, fraction) pairs "
      "that may occupy it.");
  describe_struct<atat::AtatLattice>(
      sub, "AtatLattice",
      "An ATAT rndstr.in lattice: axes, the cell in units of axes, sites, and "
      "every species label, sorted.")
      .def("occupation_index", &atat::AtatLattice::occupation_index,
           py::arg("species"), "The species' rank in labels, or -1.");
  describe_struct<atat::EnumeratedSqs>(
      sub, "EnumeratedSqs",
      "A supercell with a random start and its cluster orbits: everything a "
      "ClusterCorrelationConstraint needs (occ_index, table, orbits).");
}

void bind_functions(py::module_ &sub) {
  sub.def(
      "parse_lattice",
      [](const std::filesystem::path &path) {
        return mcsqs_call([&] { return atat::parse_lattice(path); });
      },
      py::arg("path"), "Read an ATAT rndstr.in file.");
  sub.def(
      "parse_lattice_text",
      [](const std::string &text) {
        std::istringstream in{text};
        return mcsqs_call([&] { return atat::parse_lattice(in); });
      },
      py::arg("text"), "Parse rndstr.in text.");
  sub.def(
      "write_str_out",
      [](const std::filesystem::path &path, const RMC::mat3_t &axes,
         const RMC::mat3_t &supercell, const std::vector<RMC::vec3_t> &positions,
         const std::vector<std::string> &species) {
        mcsqs_call([&] {
          atat::write_str_out(path, axes, supercell, positions, species);
        });
      },
      py::arg("path"), py::arg("axes"), py::arg("supercell"),
      py::arg("positions"), py::arg("species"),
      "Write an ATAT str.out (positions in units of axes).");
  sub.def(
      "enumerate",
      [](const atat::AtatLattice &lattice, const Eigen::Matrix3i &supercell,
         const std::map<int, double> &diameters, std::uint32_t seed) {
        const atat::Diameters by_body(diameters.begin(), diameters.end());
        return mcsqs_call([&] {
          const py::gil_scoped_release unlocked;
          return atat::enumerate(lattice, supercell, by_body, seed);
        });
      },
      py::arg("lattice"), py::arg("supercell"), py::arg("diameters"),
      py::kw_only(), py::arg("seed") = std::uint32_t{42},
      "Build the supercell (a (3, 3) integer matrix of lattice.cell) with a "
      "random start at the target composition, and enumerate cluster orbits "
      "up to diameters {body: max diameter in Angstrom}.");
}

} // namespace

void bind_mcsqs(py::module_ &m) {
  auto sub = m.def_submodule(
      "mcsqs", "The SQS (special quasirandom structure) pipeline. Import "
               "rmc.mcsqs instead.");
  bind_types(sub);
  bind_functions(sub);
  // Registered like a real submodule so `from rmc._core.mcsqs import ...`
  // and the stub generator find it.
  py::module_::import("sys").attr("modules")["rmc._core.mcsqs"] = sub;
}

} // namespace rmc::python
