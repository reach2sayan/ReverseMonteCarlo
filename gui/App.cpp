#include "App.hpp"

#include <imgui.h>
#include <imgui_internal.h> // DockBuilder*

namespace rmcgui {

bool App::init() {
  sink_ = install_gui_log_sink(log_ring_);
  return view_.init();
}

void App::dockspace() {
  const ImGuiViewport *vp = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(vp->WorkPos);
  ImGui::SetNextWindowSize(vp->WorkSize);
  ImGui::SetNextWindowViewport(vp->ID);

  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
      ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_MenuBar;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("##RMCDockHost", nullptr, flags);
  ImGui::PopStyleVar(3);

  const ImGuiID dock_id = ImGui::GetID("RMCDockSpace");
  ImGui::DockSpace(dock_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

  if (ImGui::BeginMenuBar()) {
    if (ImGui::BeginMenu("View")) {
      if (ImGui::MenuItem("Reset layout")) {
        layout_built_ = false;
      }
      ImGui::EndMenu();
    }
    ImGui::EndMenuBar();
  }

  if (!layout_built_) {
    layout_built_ = true;
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, vp->WorkSize);

    // Left (Structure) big; right column (Params over Convergence); bottom-left
    // strip (Log) — matching the requested layout.
    ImGuiID main_id = dock_id;
    const ImGuiID right_id = ImGui::DockBuilderSplitNode(
        main_id, ImGuiDir_Right, 0.32f, nullptr, &main_id);
    const ImGuiID bottom_left_id = ImGui::DockBuilderSplitNode(
        main_id, ImGuiDir_Down, 0.28f, nullptr, &main_id);
    ImGuiID right_top_id = right_id;
    const ImGuiID right_bottom_id = ImGui::DockBuilderSplitNode(
        right_top_id, ImGuiDir_Down, 0.5f, nullptr, &right_top_id);

    ImGui::DockBuilderDockWindow("Structure", main_id);
    ImGui::DockBuilderDockWindow("Log", bottom_left_id);
    ImGui::DockBuilderDockWindow("Parameters", right_top_id);
    ImGui::DockBuilderDockWindow("Convergence", right_bottom_id);
    ImGui::DockBuilderFinish(dock_id);
  }

  ImGui::End();
}

void App::frame() {
  dockspace();
  const RenderSnapshot snap = sim_.snapshot();
  draw_params_panel("Parameters", form_, sim_, snap);
  draw_structure_panel("Structure", view_, cam_, snap);
  draw_log_panel("Log", log_ring_);
  draw_convergence_panel("Convergence", sim_);
}

void App::shutdown() {
  sim_.shutdown();            // join the worker so no more log records arrive…
  remove_gui_log_sink(sink_); // …before detaching the sink it feeds.
  view_.shutdown();
}

} // namespace rmcgui
