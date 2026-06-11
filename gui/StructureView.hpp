#pragma once
#include "Camera.hpp"
#include "RenderSnapshot.hpp"

#include <glad/gl.h>
#include <cstddef>
#include <cstdint>

namespace rmcgui {

// OpenGL 3.3 renderer for the atomic structure. Draws atoms as instanced
// impostor spheres (billboarded quads + analytic sphere in the fragment shader,
// one draw call) and the cell as a wireframe, into an offscreen FBO whose colour
// texture the StructurePanel shows via ImGui::Image. Owns no engine state — it
// renders purely from a RenderSnapshot.
class StructureView {
public:
  bool init();     // compile shaders, build VAOs (needs a current GL context)
  void shutdown(); // delete all GL objects

  // Render `snap` into the FBO sized w×h, driving `cam`. Auto-frames the camera
  // the first time atoms appear (and when the atom count changes).
  void render(const RenderSnapshot &snap, Camera &cam, int w, int h);

  [[nodiscard]] GLuint texture() const { return color_tex_; }

private:
  void ensure_fbo(int w, int h);
  void upload_instances(const RenderSnapshot &snap);
  void upload_box(const RenderSnapshot &snap);

  GLuint sphere_prog_ = 0, line_prog_ = 0;
  GLuint quad_vao_ = 0, quad_vbo_ = 0, instance_vbo_ = 0;
  GLuint box_vao_ = 0, box_vbo_ = 0;
  GLuint fbo_ = 0, color_tex_ = 0, depth_rb_ = 0;
  int fb_w_ = 0, fb_h_ = 0;

  // Cached uniform locations.
  GLint u_sphere_view_ = -1, u_sphere_proj_ = -1;
  GLint u_line_view_ = -1, u_line_proj_ = -1, u_line_color_ = -1;

  std::size_t instance_cap_ = 0;   // GPU instance buffer capacity (atoms)
  std::size_t instance_count_ = 0; // atoms currently uploaded
  GLsizei box_vertex_count_ = 0;
  std::uint64_t uploaded_gen_ = 0; // snapshot generation last uploaded
  bool framed_ = false;
  std::size_t framed_natoms_ = 0;
};

} // namespace rmcgui
