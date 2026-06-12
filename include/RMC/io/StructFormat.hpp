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
  BoundaryConditions bc;
};

[[nodiscard]] Result<LoadedStructure> read_structure_by_ext(
    const std::filesystem::path &path, const BoundaryConditions &default_bc,
    const std::vector<std::string> &type_to_element = {});

} // namespace RMC::io
