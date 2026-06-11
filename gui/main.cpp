#include <glad/gl.h>
// glad must precede any GL-using header; tell GLFW not to pull its own GL.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#ifdef RMC_GUI_USE_IMPLOT
#include <implot.h>
#endif

#include "App.hpp"

#include <boost/log/trivial.hpp>

namespace {
void glfw_error(int code, const char *desc) {
  BOOST_LOG_TRIVIAL(error) << "GLFW error " << code << ": " << desc;
}
} // namespace

int main() {
  glfwSetErrorCallback(glfw_error);
  if (!glfwInit()) {
    BOOST_LOG_TRIVIAL(error) << "glfwInit failed";
    return 1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  GLFWwindow *window =
      glfwCreateWindow(1440, 900, "RMC — Reverse Monte Carlo", nullptr, nullptr);
  if (window == nullptr) {
    BOOST_LOG_TRIVIAL(error) << "glfwCreateWindow failed";
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // vsync

  if (gladLoadGL(glfwGetProcAddress) == 0) {
    BOOST_LOG_TRIVIAL(error) << "gladLoadGL failed";
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
#ifdef RMC_GUI_USE_IMPLOT
  ImPlot::CreateContext();
#endif
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  {
    rmcgui::App app;
    if (!app.init()) {
      BOOST_LOG_TRIVIAL(error) << "GUI init failed (shader/FBO setup)";
    }

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      app.frame();

      ImGui::Render();
      int dw = 0, dh = 0;
      glfwGetFramebufferSize(window, &dw, &dh);
      glViewport(0, 0, dw, dh);
      glClearColor(0.10f, 0.10f, 0.12f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      glfwSwapBuffers(window);
    }

    app.shutdown(); // tear down GL + worker while the context is still current
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
#ifdef RMC_GUI_USE_IMPLOT
  ImPlot::DestroyContext();
#endif
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
