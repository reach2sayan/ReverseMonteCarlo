#pragma once
#include <RMC/Engine.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>

#include <boost/program_options.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace RMC {

// =====================================================================
// Reusable simulation setup — the program_options-independent core that
// turns "inputs" into a ready Engine. RMC_run maps its command line onto
// RMCConfig, which feeds the single build path (build_engine).
// =====================================================================

// Move-proposal strategy. Random is the classic RMC walk; Langevin (MALA) and
// Leapfrog (HMC) steer atoms along −∇χ² of the constraints toward the target,
// converging far faster but costing ~6k constraint evals per move for the
// finite-difference gradient.
enum class MoveGenKind { Random, Langevin, Leapfrog };

// Plain inputs mirroring the CLI knobs.
struct RMCConfig {
  // ---- input structure (exactly one of these paths non-empty) ----
  std::string pdb_path;
  std::string lammps_path;
  std::string vasp_path;
  std::vector<std::string> lammps_types;   // 'Zr Cu Ag' type→element legend
  std::optional<std::string> box_override; // "a b c" orthogonal, or "inf"

  // ---- experimental targets (engaged optional = supplied) ----
  std::optional<std::string> pdf_path;
  std::optional<std::string> sq_path;
  std::optional<std::string> adf_path;

  // ---- numeric parameters ----
  double rho0 = 0.1;
  double adf_cutoff = 3.4;
  int adf_smooth = 2;
  std::uint64_t steps = 100000;
  std::uint32_t seed = 42;
  bool use_smart = false;
  double group_min_amp = 0.0;
  double group_max_amp = 0.2;
  MoveGenKind move_gen = MoveGenKind::Random; // CLI default keeps the classic walk
  double move_step = 0.05; // ε (Å) for Langevin/Leapfrog gradient moves
  std::optional<std::string> checkpoint_path;
  std::uint64_t log_every = 1000;

  std::string out_path = "refined.pdb";
};

struct LoadedStructure {
  AtomicStructure structure;
  BoundaryConditions bc = InfiniteBC(1.0);
};

struct ExperimentalData {
  std::optional<mat_t> pdf, sq, adf;
};

[[nodiscard]] Result<LoadedStructure> load_structure(const RMCConfig &cfg);

[[nodiscard]] Result<ExperimentalData>
load_experimental_data(const RMCConfig &cfg);

void attach_constraints(Engine &engine, const ExperimentalData &data,
                        const RMCConfig &cfg);

// Assemble an engine from ALREADY-loaded inputs: Engine(s,bc) (copying the
// structure, so each call owns its own) → build_atomic_groups → optional smart
// selector → attach constraints. The CLI ensemble factory uses this to build N
// replicas from one disk read; vary cfg.seed per replica.
[[nodiscard]] Engine build_engine(const LoadedStructure &loaded,
                                  const ExperimentalData &data,
                                  const RMCConfig &cfg);

// Full path: load structure + experimental data from disk, then build_engine().
// Returns a ready (not yet run) engine by move. Checkpoint/step-callback that
// are run-policy rather than build inputs are left to the caller.
[[nodiscard]] Result<Engine> build_engine(const RMCConfig &cfg);

// Apply cfg.move_gen to an already-built engine whose constraints are attached
// and which is in its FINAL location (the gradient generators bind a pointer to
// engine.constraints(), which moving the engine afterwards would invalidate).
// Random is a no-op (build_engine already made the random-walk groups);
// Langevin/Leapfrog rebuild the groups with the gradient proposer.
void apply_move_generator(Engine &engine, const RMCConfig &cfg);

[[nodiscard]] mat3_t periodic_box_or_zero(const BoundaryConditions &bc);

// Write a structure, picking the format from the output path's extension
// (.vasp/.poscar → VASP, .lammps/.lmp/.data → LAMMPS, else PDB). `box` comes
// from the active periodic cell (zero for infinite). Shared by the CLI's final
// write and the GUI's end-of-run save.
[[nodiscard]] Result<void> write_structure_by_ext(const AtomicStructure &s,
                                                  const mat3_t &box,
                                                  const std::string &path);

class RMCRunner {
public:
  RMCRunner();
  int run(int argc, char **argv);
private:
  boost::program_options::options_description options_;
};

} // namespace RMC
