#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace RMC::io {

// Per-atom column layout of the `Atoms` section. LAMMPS does not record the
// atom_style in the data file, so the caller must say how many leading columns
// precede the x,y,z coordinates. The default (Atomic) matches files written by
// `atom_style atomic`: `atom-id atom-type x y z`.
//   Atomic    : id type x y z
//   Charge    : id type q x y z
//   Molecular : id mol type x y z
//   Full      : id mol type q x y z
enum class LammpsAtomStyle { Atomic, Charge, Molecular, Full };

// Result of reading a LAMMPS data file: the atoms plus the simulation box the
// file declares. `box` holds the cell edge vectors as columns (triclinic tilt
// factors xy/xz/yz are honoured); `origin` is the lower corner (xlo,ylo,zlo).
struct LammpsData {
  AtomicStructure structure;
  mat3_t box{mat3_t::Zero()};
  vec3_t origin{vec3_t::Zero()};

  // Convenience: build a PeriodicBC from the parsed box.
  [[nodiscard]] PeriodicBC periodic_bc() const { return PeriodicBC(box); }
};

// Parse a LAMMPS data file (header + `Atoms` section). Coordinates are stored
// as the absolute Cartesian values found in the file.
//
// LAMMPS atom *types* are integers with no intrinsic element identity. Pass
// `type_to_element` to label them: index 0 -> type 1, index 1 -> type 2, etc.
// (e.g. {"Zr","Cu","Ag"}). Recognised symbols get their atomic number; the
// element string is also stored on every atom. When the mapping is empty (or
// shorter than the number of types) the unmapped types fall back to element
// "X<type>" with atomic_number = type, which still keeps species distinct.
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
