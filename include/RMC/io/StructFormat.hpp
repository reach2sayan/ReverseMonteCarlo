#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace RMC::io {

enum class StructFormat { Pdb, Vasp, Lammps };
[[nodiscard]] std::optional<StructFormat>
classify_structure_format(const std::filesystem::path &p);

struct LoadedStructure {
  AtomicStructure structure;
  BoundaryConditions bc = InfiniteBC(1.0);
};

// Read a structure in the given format. VASP and LAMMPS files carry their own
// periodic cell; a PDB gets `default_bc`. `type_to_element` names LAMMPS types.
[[nodiscard]] Result<LoadedStructure>
read_structure(const std::filesystem::path &path, StructFormat fmt,
               const BoundaryConditions &default_bc,
               const std::vector<std::string> &type_to_element = {});

// read_structure with the format taken from the path (unknown ⇒ LAMMPS).
[[nodiscard]] Result<LoadedStructure> read_structure_by_ext(
    const std::filesystem::path &path, const BoundaryConditions &default_bc,
    const std::vector<std::string> &type_to_element = {});

} // namespace RMC::io
