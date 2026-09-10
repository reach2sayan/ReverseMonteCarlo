#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace RMC::io {

// Per-atom column layout of the `Atoms` section (the data file doesn't record
// atom_style, so the caller picks it). Default Atomic matches `atom_style atomic`.
//   Atomic    : id type x y z
//   Charge    : id type q x y z
//   Molecular : id mol type x y z
//   Full      : id mol type q x y z
enum class LammpsAtomStyle { Atomic, Charge, Molecular, Full };

// Atoms plus declared box. `box` holds cell edge vectors as columns (triclinic
// tilt xy/xz/yz honoured); `origin` is the lower corner (xlo,ylo,zlo).
struct LammpsData {
  AtomicStructure structure;
  mat3_t box{mat3_t::Zero()};
  vec3_t origin{vec3_t::Zero()};

  // Convenience: build a PeriodicBC from the parsed box.
  [[nodiscard]] PeriodicBC periodic_bc() const { return PeriodicBC(box); }
};

// Parse a LAMMPS data file (header + `Atoms`); coordinates are absolute Cartesian.
// `type_to_element` labels integer atom types: index 0 -> type 1, etc.
// (e.g. {"Zr","Cu","Ag"}). Unmapped types fall back to "X<type>" / atomic_number=type.
[[nodiscard]] Result<LammpsData>
read_lammps_data(const std::filesystem::path &path,
                 const std::vector<std::string> &type_to_element = {},
                 LammpsAtomStyle style = LammpsAtomStyle::Atomic);

// Write an orthorhombic LAMMPS data file (atom_style atomic) from a structure
// and box. Atom types are assigned by first appearance of each element symbol;
// the type->element legend is emitted as a trailing comment.
[[nodiscard]] Result<void>
write_lammps_data(const AtomicStructure &s, const mat3_t &box,
                  const std::filesystem::path &path,
                  const vec3_t &origin = vec3_t::Zero());

} // namespace RMC::io
