#include "Panels.hpp"

#include <imgui.h>
#include <imgui_stdlib.h>
#ifdef RMC_GUI_USE_IMPLOT
#include <implot.h>
#else
#include <cfloat>
#endif

#include <cstdint>
#include <sstream>
#include <vector>

namespace rmcgui {

namespace {

const char *state_label(RunState s) {
  switch (s) {
  case RunState::Idle:
    return "idle";
  case RunState::Building:
    return "building…";
  case RunState::Running:
    return "running";
  case RunState::Paused:
    return "paused";
  case RunState::Finished:
    return "finished";
  case RunState::Error:
    return "error";
  }
  return "?";
}

ImVec4 severity_color(LogSeverity s) {
  switch (s) {
  case LogSeverity::Trace:
  case LogSeverity::Debug:
    return ImVec4(0.55f, 0.55f, 0.55f, 1.0f);
  case LogSeverity::Info:
    return ImVec4(0.85f, 0.86f, 0.88f, 1.0f);
  case LogSeverity::Warning:
    return ImVec4(0.95f, 0.80f, 0.30f, 1.0f);
  case LogSeverity::Error:
  case LogSeverity::Fatal:
    return ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
  }
  return ImVec4(1, 1, 1, 1);
}

} // namespace

RMC::RMCConfig ParamsForm::to_config() const {
  RMC::RMCConfig c;
  switch (input_kind) {
  case 1: {
    c.lammps_path = lammps_path;
    std::istringstream is(lammps_types);
    for (std::string t; is >> t;) {
      c.lammps_types.push_back(t);
    }
    break;
  }
  case 2:
    c.vasp_path = vasp_path;
    break;
  default:
    c.pdb_path = pdb_path;
    break;
  }
  if (override_box && !box.empty()) {
    c.box_override = box;
  }
  if (!pdf_path.empty()) {
    c.pdf_path = pdf_path;
  }
  if (!sq_path.empty()) {
    c.sq_path = sq_path;
  }
  if (!adf_path.empty()) {
    c.adf_path = adf_path;
  }
  c.rho0 = rho0;
  c.adf_cutoff = adf_cutoff;
  c.adf_smooth = adf_smooth;
  c.steps = static_cast<std::uint64_t>(steps < 0 ? 0 : steps);
  c.seed = static_cast<std::uint32_t>(seed < 0 ? 0 : seed);
  c.use_smart = use_smart;
  c.log_every = static_cast<std::uint64_t>(log_every < 1 ? 1 : log_every);
  return c;
}

void draw_params_panel(const char *title, ParamsForm &form, SimController &sim,
                       const RenderSnapshot &snap) {
  if (ImGui::Begin(title)) {
    const char *kinds[] = {"PDB", "LAMMPS", "VASP"};
    ImGui::Combo("Input", &form.input_kind, kinds, IM_ARRAYSIZE(kinds));
    if (form.input_kind == 0) {
      ImGui::InputText("PDB path", &form.pdb_path);
    } else if (form.input_kind == 1) {
      ImGui::InputText("LAMMPS data", &form.lammps_path);
      ImGui::InputText("Types", &form.lammps_types);
    } else {
      ImGui::InputText("VASP POSCAR", &form.vasp_path);
    }
    ImGui::Checkbox("Override box", &form.override_box);
    if (form.override_box) {
      ImGui::InputText("Box (a b c | inf)", &form.box);
    }

    ImGui::SeparatorText("Targets (optional)");
    ImGui::InputText("G(r)", &form.pdf_path);
    ImGui::InputText("S(Q)", &form.sq_path);
    ImGui::InputText("ADF", &form.adf_path);

    ImGui::SeparatorText("Parameters");
    ImGui::InputDouble("rho0", &form.rho0, 0.0, 0.0, "%.4f");
    ImGui::InputDouble("ADF cutoff", &form.adf_cutoff, 0.0, 0.0, "%.3f");
    ImGui::InputInt("ADF smooth", &form.adf_smooth);
    ImGui::InputInt("Steps", &form.steps);
    ImGui::InputInt("Seed", &form.seed);
    ImGui::Checkbox("Smart selector", &form.use_smart);
    ImGui::InputInt("Refresh every", &form.log_every);

    ImGui::Separator();
    const RunState st = sim.state();
    const bool busy = sim.active();
    if (!busy) {
      if (ImGui::Button("Start")) {
        sim.start(form.to_config());
      }
    } else {
      if (st == RunState::Paused) {
        if (ImGui::Button("Resume")) {
          sim.resume();
        }
      } else if (ImGui::Button("Pause")) {
        sim.pause();
      }
      ImGui::SameLine();
      if (ImGui::Button("Stop")) {
        sim.stop();
      }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("[%s]", state_label(st));

    if (st == RunState::Error) {
      ImGui::PushStyleColor(ImGuiCol_Text, severity_color(LogSeverity::Error));
      ImGui::TextWrapped("%s", sim.last_error().c_str());
      ImGui::PopStyleColor();
    }

    ImGui::Separator();
    const double rate =
        snap.tried > 0
            ? 100.0 * static_cast<double>(snap.accepted) /
                  static_cast<double>(snap.tried)
            : 0.0;
    ImGui::Text("atoms: %zu", snap.natoms);
    ImGui::Text("step:  %llu", static_cast<unsigned long long>(snap.step));
    ImGui::Text("accept: %.1f%%  (%llu/%llu)", rate,
                static_cast<unsigned long long>(snap.accepted),
                static_cast<unsigned long long>(snap.tried));
    ImGui::Text("chi2:  %.6g", snap.chi2);
  }
  ImGui::End();
}

void draw_structure_panel(const char *title, StructureView &view, Camera &cam,
                          const RenderSnapshot &snap) {
  if (ImGui::Begin(title)) {
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const int w = static_cast<int>(avail.x);
    const int h = static_cast<int>(avail.y);
    view.render(snap, cam, w, h);
    // Flip V: GL textures are bottom-up, ImGui expects top-down. The C-style
    // cast is the portable ImGui idiom (works whether ImTextureID is void* or
    // an integer typedef).
    ImGui::Image((ImTextureID)(std::intptr_t)view.texture(), avail,
                 ImVec2(0, 1), ImVec2(1, 0));
    if (ImGui::IsItemHovered()) {
      ImGuiIO &io = ImGui::GetIO();
      if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
        cam.orbit(io.MouseDelta.x, io.MouseDelta.y);
      }
      if (ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f)) {
        cam.pan(io.MouseDelta.x, io.MouseDelta.y);
      }
      if (io.MouseWheel != 0.0f) {
        cam.zoom(io.MouseWheel);
      }
    }
  }
  ImGui::End();
}

void draw_log_panel(const char *title, LogRing &ring) {
  if (ImGui::Begin(title)) {
    static bool autoscroll = true;
    ImGui::Checkbox("Auto-scroll", &autoscroll);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
      ring.clear();
    }
    ImGui::Separator();
    ImGui::BeginChild("loglines", ImVec2(0, 0), false,
                      ImGuiWindowFlags_HorizontalScrollbar);
    for (const LogLine &line : ring.snapshot()) {
      ImGui::PushStyleColor(ImGuiCol_Text, severity_color(line.sev));
      ImGui::TextUnformatted(line.text.c_str());
      ImGui::PopStyleColor();
    }
    if (autoscroll) {
      ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
  }
  ImGui::End();
}

void draw_convergence_panel(const char *title, SimController &sim) {
  if (ImGui::Begin(title)) {
    std::vector<float> xs, ys;
    sim.convergence(xs, ys);
#ifdef RMC_GUI_USE_IMPLOT
    if (ImPlot::BeginPlot("##chi2", ImVec2(-1, 200))) {
      ImPlot::SetupAxes("step", "chi2", ImPlotAxisFlags_AutoFit,
                        ImPlotAxisFlags_AutoFit);
      if (!xs.empty()) {
        ImPlot::PlotLine("chi2", xs.data(), ys.data(),
                         static_cast<int>(xs.size()));
      }
      ImPlot::EndPlot();
    }
#else
    if (!ys.empty()) {
      ImGui::PlotLines("##chi2", ys.data(), static_cast<int>(ys.size()), 0,
                       "chi2", FLT_MAX, FLT_MAX, ImVec2(-1, 200));
    }
#endif
    ImGui::SeparatorText("Per-constraint error");
    if (ImGui::BeginTable("breakdown", 2,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
      ImGui::TableSetupColumn("constraint");
      ImGui::TableSetupColumn("chi2");
      ImGui::TableHeadersRow();
      for (const auto &[name, val] : sim.breakdown()) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(name.c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.6g", val);
      }
      ImGui::EndTable();
    }
  }
  ImGui::End();
}

} // namespace rmcgui
