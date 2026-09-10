#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
#include <filesystem>

namespace RMC::io {

// Result of reading a VASP POSCAR/CONTCAR: the atoms plus the cell the file
// declares. `box` holds the cell edge vectors as columns (the three lattice
// vectors a1,a2,a3 — general triclinic cells are supported). Coordinates are
// stored as absolute Cartesian values.
struct VaspData {
  AtomicStructure structure;
  mat3_t box{mat3_t::Zero()};

  [[nodiscard]] PeriodicBC periodic_bc() const { return PeriodicBC(box); }
};

// Parse a VASP5 POSCAR/CONTCAR file:
//   line 1     comment
//   line 2     universal scaling factor (negative ⇒ target volume)
//   lines 3-5  lattice vectors a1, a2, a3 (one per line)
//   line 6     element symbols (VASP5)
//   line 7     atom count per element
//   [optional] "Selective dynamics"
//   next       coordinate mode: Direct/fractional or Cartesian
//   then       Σcounts coordinate lines (trailing flags ignored)
//
// VASP4 files (no element-symbol line) are rejected with a clear error, since
// the species cannot be recovered. Element symbols are mapped to atomic numbers
// via seitz::data; an unrecognised symbol keeps atomic_number 0.
[[nodiscard]] Result<VaspData> read_vasp(const std::filesystem::path &path);

// Write a VASP5 POSCAR (scaling factor 1.0, Direct coordinates). Atoms are
// grouped by element in first-appearance order (the format requires grouping),
// so the element/count lines and the coordinate block stay consistent. The box
// columns are written as the three lattice-vector rows.
[[nodiscard]] Result<void> write_vasp(const AtomicStructure &s,
                                      const mat3_t &box,
                                      const std::filesystem::path &path);

} // namespace RMC::io
