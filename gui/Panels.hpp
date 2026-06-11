#pragma once
#include "Camera.hpp"
#include "LogSink.hpp"
#include "RenderSnapshot.hpp"
#include "SimController.hpp"
#include "StructureView.hpp"

#include <RMC/RMCRunner.hpp>

#include <string>

namespace rmcgui {

// Mutable state bound to the Parameters-panel widgets. to_config() converts it
// into the RMC::RMCConfig the shared builder consumes.
struct ParamsForm {
  int input_kind = 0; // 0 = PDB, 1 = LAMMPS, 2 = VASP
  std::string pdb_path = "bestsqs.pdb";
  std::string lammps_path;
  std::string lammps_types; // e.g. "Zr Cu"
  std::string vasp_path;
  bool override_box = false;
  std::string box = "20 20 20"; // "a b c" or "inf"

  std::string pdf_path;
  std::string sq_path;
  std::string adf_path;

  double rho0 = 0.1;
  double adf_cutoff = 3.4;
  int adf_smooth = 2;
  int steps = 100000;
  int seed = 42;
  bool use_smart = false;
  int log_every = 100; // render-refresh cadence (step-callback interval)

  [[nodiscard]] RMC::RMCConfig to_config() const;
};

void draw_params_panel(const char *title, ParamsForm &form, SimController &sim,
                       const RenderSnapshot &snap);
void draw_structure_panel(const char *title, StructureView &view, Camera &cam,
                          const RenderSnapshot &snap);
void draw_log_panel(const char *title, LogRing &ring);
void draw_convergence_panel(const char *title, SimController &sim);

} // namespace rmcgui
