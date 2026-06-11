#pragma once
#include "Camera.hpp"
#include "LogSink.hpp"
#include "Panels.hpp"
#include "SimController.hpp"
#include "StructureView.hpp"

#include <boost/shared_ptr.hpp>

namespace rmcgui {

// Top-level GUI application: owns the simulation controller, the GL structure
// renderer, the log ring + its Boost.Log sink, and the panel form state. Drives
// one ImGui frame (dockspace + four panels) per call to frame().
class App {
public:
  bool init();     // install log sink + build GL resources (context must exist)
  void frame();    // one ImGui frame
  void shutdown(); // join worker, remove sink, free GL resources

private:
  void dockspace();

  SimController sim_;
  StructureView view_;
  Camera cam_;
  LogRing log_ring_;
  boost::shared_ptr<GuiLogSink> sink_;
  ParamsForm form_;
  bool layout_built_ = false;
};

} // namespace rmcgui
