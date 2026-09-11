#include "casters.hpp"
#include "describe.hpp"
#include "errors.hpp"

#include <RMC/io/Checkpoint.hpp>
#include <RMC/io/DataReader.hpp>
#include <RMC/io/LammpsReader.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/io/StructFormat.hpp>
#include <RMC/io/VaspReader.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <filesystem>
#include <format>
#include <string>
#include <vector>

// Structure, data and checkpoint files. Every path parameter takes str or any
// os.PathLike (pathlib.Path); every failure raises IoError.
namespace rmc::python {

namespace {

namespace io = RMC::io;
using RMC::AtomicStructure;

constexpr auto kIo = &ErrorTypes::io;

// A checkpoint holds coordinates only. Load it beside the structure, then copy
// in place, so a file for a different atom count is refused instead of
// resizing an array that constraints hold spans into.
io::EngineStats load_checkpoint(AtomicStructure &s,
                                const std::filesystem::path &path) {
  AtomicStructure scratch;
  const io::EngineStats stats = unwrap_nogil(
      error_types().io, [&] { return io::load_checkpoint(scratch, path); });
  if (scratch.size() != s.size()) {
    detail::raise(error_types().io,
                  std::format("checkpoint {} holds {} atoms; the structure has {}",
                              path.string(), scratch.size(), s.size())
                      .c_str());
  }
  s.coordinates = scratch.coordinates;
  return stats;
}

void bind_results(py::module_ &m) {
  describe_struct<io::EngineStats>(
      m, "EngineStats",
      "Step counters and the last total chi^2, as a checkpoint stores them.");
  bind_structure_result<io::LoadedStructure>(
      m, "LoadedStructure",
      "A structure and the boundary conditions its file implied.");
  bind_structure_result<io::VaspData>(m, "VaspData",
                                      "A POSCAR/CONTCAR structure and its cell.");
  bind_structure_result<io::LammpsData>(
      m, "LammpsData", "A LAMMPS data-file structure, its cell and lower corner.");
}

void bind_files(py::module_ &m) {
  const std::vector<std::string> no_legend;
  const RMC::InfiniteBC open_space{1.0};

  m.def("read_pdb", checked<&io::read_pdb>(kIo), py::arg("path"),
        "Read a PDB file (ATOM/HETATM records).");
  m.def("write_pdb", checked<&io::write_pdb>(kIo), py::arg("structure"),
        py::arg("path"), "Write a PDB file.");
  m.def("read_vasp", checked<&io::read_vasp>(kIo), py::arg("path"),
        "Read a VASP5 POSCAR/CONTCAR (element line required).");
  m.def("write_vasp", checked<&io::write_vasp>(kIo), py::arg("structure"),
        py::arg("box"), py::arg("path"),
        "Write a VASP5 POSCAR; box columns are the cell vectors.");
  m.def("read_lammps_data", checked<&io::read_lammps_data>(kIo),
        py::arg("path"), py::arg("type_to_element") = no_legend, py::kw_only(),
        py::arg("style") = io::LammpsAtomStyle::Atomic,
        "Read a LAMMPS data file. type_to_element names atom types 1, 2, ... "
        "in order; unnamed types keep their number as the element.");
  m.def("write_lammps_data", checked<&io::write_lammps_data>(kIo),
        py::arg("structure"), py::arg("box"), py::arg("path"), py::kw_only(),
        py::arg("origin") = RMC::vec3_t::Zero().eval(),
        "Write a LAMMPS data file (atom_style atomic); types are numbered by "
        "first appearance of each element, with a legend comment.");
  m.def("read_structure", checked<&io::read_structure>(kIo), py::arg("path"),
        py::arg("fmt"), py::kw_only(), py::arg("default_bc") = open_space,
        py::arg("type_to_element") = no_legend,
        "Read a structure in the given format. Formats that carry a cell "
        "(VASP, LAMMPS) supply the boundary conditions; PDB gets default_bc.");
  m.def("read_structure_by_ext", checked<&io::read_structure_by_ext>(kIo),
        py::arg("path"), py::kw_only(), py::arg("default_bc") = open_space,
        py::arg("type_to_element") = no_legend,
        "read_structure with the format taken from the file name (unknown "
        "names are read as LAMMPS).");
  m.def("classify_structure_format", &io::classify_structure_format,
        py::arg("path"),
        "The format a file name implies (.pdb; .vasp/.poscar or "
        "POSCAR/CONTCAR; .lammps/.lmp/.data), or None.");
  m.def("read_xy_data", checked<&io::read_xy_data>(kIo), py::arg("path"),
        "Read two-column numeric data (e.g. r, G(r)) as an (N, 2) array.");
  m.def("read_columns", checked<&io::read_columns>(kIo), py::arg("path"),
        "Read multi-column numeric data (e.g. an ADF table) as an (N, k) "
        "array.");
  m.def("save_checkpoint", checked<&io::save_checkpoint>(kIo),
        py::arg("structure"), py::arg("stats"), py::arg("path"),
        "Save coordinates and step counters to a binary checkpoint.");
  m.def("load_checkpoint", &load_checkpoint, py::arg("structure"),
        py::arg("path"),
        "Load a checkpoint's coordinates into structure, in place, and return "
        "its step counters. Refuses a checkpoint for a different atom count.");
}

} // namespace

void bind_io(py::module_ &m) {
  bind_results(m);
  bind_files(m);
}

} // namespace rmc::python
