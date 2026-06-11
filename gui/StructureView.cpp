#include "StructureView.hpp"

#include "ElementColors.hpp"

#include <Eigen/Core>
#include <boost/log/trivial.hpp>

#include <array>
#include <limits>
#include <vector>

namespace rmcgui {

namespace {

constexpr const char *kSphereVert = R"(#version 330 core
layout(location=0) in vec2 aCorner;   // quad corner in [-1,1]
layout(location=1) in vec3 aCenter;   // per-instance
layout(location=2) in float aRadius;  // per-instance
layout(location=3) in vec3 aColor;    // per-instance
uniform mat4 uView;
uniform mat4 uProj;
out vec2 vCorner;
out vec3 vColor;
out vec3 vViewCenter;
out float vRadius;
void main() {
  vec4 viewCenter = uView * vec4(aCenter, 1.0);
  vViewCenter = viewCenter.xyz;
  vColor = aColor;
  vRadius = aRadius;
  vCorner = aCorner;
  vec3 viewPos = viewCenter.xyz + vec3(aCorner * aRadius, 0.0);
  gl_Position = uProj * vec4(viewPos, 1.0);
}
)";

constexpr const char *kSphereFrag = R"(#version 330 core
in vec2 vCorner;
in vec3 vColor;
in vec3 vViewCenter;
in float vRadius;
uniform mat4 uProj;
out vec4 fragColor;
void main() {
  float r2 = dot(vCorner, vCorner);
  if (r2 > 1.0) discard;
  float z = sqrt(1.0 - r2);
  vec3 normal = vec3(vCorner, z);
  // View-space point on the sphere surface, for correct per-pixel depth.
  vec3 viewPos = vViewCenter + vec3(vCorner * vRadius, z * vRadius);
  vec4 clip = uProj * vec4(viewPos, 1.0);
  gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;
  vec3 lightDir = normalize(vec3(0.35, 0.45, 1.0));
  float diff = max(dot(normal, lightDir), 0.0);
  vec3 color = vColor * (0.25 + 0.75 * diff);
  fragColor = vec4(color, 1.0);
}
)";

constexpr const char *kLineVert = R"(#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uView;
uniform mat4 uProj;
void main() { gl_Position = uProj * uView * vec4(aPos, 1.0); }
)";

constexpr const char *kLineFrag = R"(#version 330 core
uniform vec3 uColor;
out vec4 fragColor;
void main() { fragColor = vec4(uColor, 1.0); }
)";

GLuint compile_shader(GLenum type, const char *src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(s, sizeof(log), nullptr, log);
    BOOST_LOG_TRIVIAL(error) << "StructureView shader compile failed: " << log;
  }
  return s;
}

GLuint link_program(const char *vs, const char *fs) {
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs);
  GLuint p = glCreateProgram();
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetProgramInfoLog(p, sizeof(log), nullptr, log);
    BOOST_LOG_TRIVIAL(error) << "StructureView program link failed: " << log;
  }
  glDeleteShader(v);
  glDeleteShader(f);
  return p;
}

} // namespace

bool StructureView::init() {
  sphere_prog_ = link_program(kSphereVert, kSphereFrag);
  line_prog_ = link_program(kLineVert, kLineFrag);
  u_sphere_view_ = glGetUniformLocation(sphere_prog_, "uView");
  u_sphere_proj_ = glGetUniformLocation(sphere_prog_, "uProj");
  u_line_view_ = glGetUniformLocation(line_prog_, "uView");
  u_line_proj_ = glGetUniformLocation(line_prog_, "uProj");
  u_line_color_ = glGetUniformLocation(line_prog_, "uColor");

  // Unit quad (triangle strip) shared by every sphere instance.
  constexpr float quad[8] = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
  glGenVertexArrays(1, &quad_vao_);
  glBindVertexArray(quad_vao_);
  glGenBuffers(1, &quad_vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);

  // Per-instance buffer: center(3) radius(1) color(3) = 7 floats, stride 28.
  glGenBuffers(1, &instance_vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, instance_vbo_);
  constexpr GLsizei kStride = 7 * sizeof(float);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, kStride, nullptr);
  glVertexAttribDivisor(1, 1);
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, kStride,
                        reinterpret_cast<void *>(3 * sizeof(float)));
  glVertexAttribDivisor(2, 1);
  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, kStride,
                        reinterpret_cast<void *>(4 * sizeof(float)));
  glVertexAttribDivisor(3, 1);

  // Box wireframe buffer (dynamic, up to 24 vertices).
  glGenVertexArrays(1, &box_vao_);
  glBindVertexArray(box_vao_);
  glGenBuffers(1, &box_vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, box_vbo_);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  return sphere_prog_ != 0 && line_prog_ != 0;
}

void StructureView::ensure_fbo(int w, int h) {
  if (w == fb_w_ && h == fb_h_ && fbo_ != 0) {
    return;
  }
  if (fbo_ == 0) {
    glGenFramebuffers(1, &fbo_);
  }
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

  if (color_tex_ == 0) {
    glGenTextures(1, &color_tex_);
  }
  glBindTexture(GL_TEXTURE_2D, color_tex_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         color_tex_, 0);

  if (depth_rb_ == 0) {
    glGenRenderbuffers(1, &depth_rb_);
  }
  glBindRenderbuffer(GL_RENDERBUFFER, depth_rb_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, depth_rb_);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  fb_w_ = w;
  fb_h_ = h;
}

void StructureView::upload_instances(const RenderSnapshot &snap) {
  const std::size_t n = snap.natoms;
  std::vector<float> data;
  data.reserve(7 * n);
  for (std::size_t i = 0; i < n; ++i) {
    const ElementStyle st = element_style(snap.z[i]);
    data.push_back(snap.xyz[3 * i + 0]);
    data.push_back(snap.xyz[3 * i + 1]);
    data.push_back(snap.xyz[3 * i + 2]);
    data.push_back(st.radius);
    data.push_back(st.color[0]);
    data.push_back(st.color[1]);
    data.push_back(st.color[2]);
  }
  glBindBuffer(GL_ARRAY_BUFFER, instance_vbo_);
  const GLsizeiptr bytes =
      static_cast<GLsizeiptr>(data.size() * sizeof(float));
  if (n > instance_cap_) {
    glBufferData(GL_ARRAY_BUFFER, bytes, data.data(), GL_DYNAMIC_DRAW);
    instance_cap_ = n;
  } else if (n > 0) {
    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, data.data());
  }
  instance_count_ = n;
}

void StructureView::upload_box(const RenderSnapshot &snap) {
  const auto &b = snap.box;
  bool any = false;
  for (float v : b) {
    any = any || (v != 0.0f);
  }
  if (!any) {
    box_vertex_count_ = 0;
    return;
  }
  const Eigen::Vector3f a(b[0], b[1], b[2]);
  const Eigen::Vector3f bb(b[3], b[4], b[5]);
  const Eigen::Vector3f c(b[6], b[7], b[8]);
  const std::array<Eigen::Vector3f, 8> corner = {
      Eigen::Vector3f::Zero(), a,      bb,         c,
      a + bb,                  a + c,  bb + c,     a + bb + c};
  static constexpr int edges[12][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 4},
                                       {1, 5}, {2, 4}, {2, 6}, {3, 5},
                                       {3, 6}, {4, 7}, {5, 7}, {6, 7}};
  std::array<float, 24 * 3> verts{};
  int k = 0;
  for (const auto &e : edges) {
    for (int side = 0; side < 2; ++side) {
      const Eigen::Vector3f &p = corner[static_cast<std::size_t>(e[side])];
      verts[static_cast<std::size_t>(k++)] = p.x();
      verts[static_cast<std::size_t>(k++)] = p.y();
      verts[static_cast<std::size_t>(k++)] = p.z();
    }
  }
  glBindBuffer(GL_ARRAY_BUFFER, box_vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts.data(), GL_DYNAMIC_DRAW);
  box_vertex_count_ = 24;
}

void StructureView::render(const RenderSnapshot &snap, Camera &cam, int w,
                           int h) {
  w = w < 1 ? 1 : w;
  h = h < 1 ? 1 : h;
  ensure_fbo(w, h);

  // Auto-frame the camera the first time atoms appear / when the count changes.
  if (snap.generation != 0 && snap.natoms > 0 &&
      (!framed_ || framed_natoms_ != snap.natoms)) {
    Eigen::Vector3f mn(std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max());
    Eigen::Vector3f mx = -mn;
    for (std::size_t i = 0; i < snap.natoms; ++i) {
      const Eigen::Vector3f p(snap.xyz[3 * i + 0], snap.xyz[3 * i + 1],
                              snap.xyz[3 * i + 2]);
      mn = mn.cwiseMin(p);
      mx = mx.cwiseMax(p);
    }
    const Eigen::Vector3f center = 0.5f * (mn + mx);
    float radius = 0.5f * (mx - mn).norm();
    if (radius < 1e-3f) {
      radius = 1.0f;
    }
    cam.frame(center, radius);
    framed_ = true;
    framed_natoms_ = snap.natoms;
  }

  if (snap.generation != uploaded_gen_) {
    upload_instances(snap);
    upload_box(snap);
    uploaded_gen_ = snap.generation;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glViewport(0, 0, fb_w_, fb_h_);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  const Eigen::Matrix4f view = cam.view();
  const Eigen::Matrix4f proj =
      cam.proj(static_cast<float>(fb_w_) / static_cast<float>(fb_h_));

  if (box_vertex_count_ > 0) {
    glUseProgram(line_prog_);
    glUniformMatrix4fv(u_line_view_, 1, GL_FALSE, view.data());
    glUniformMatrix4fv(u_line_proj_, 1, GL_FALSE, proj.data());
    glUniform3f(u_line_color_, 0.45f, 0.48f, 0.55f);
    glBindVertexArray(box_vao_);
    glDrawArrays(GL_LINES, 0, box_vertex_count_);
  }

  if (instance_count_ > 0) {
    glUseProgram(sphere_prog_);
    glUniformMatrix4fv(u_sphere_view_, 1, GL_FALSE, view.data());
    glUniformMatrix4fv(u_sphere_proj_, 1, GL_FALSE, proj.data());
    glBindVertexArray(quad_vao_);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          static_cast<GLsizei>(instance_count_));
  }

  glBindVertexArray(0);
  glUseProgram(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void StructureView::shutdown() {
  glDeleteProgram(sphere_prog_);
  glDeleteProgram(line_prog_);
  glDeleteVertexArrays(1, &quad_vao_);
  glDeleteVertexArrays(1, &box_vao_);
  glDeleteBuffers(1, &quad_vbo_);
  glDeleteBuffers(1, &instance_vbo_);
  glDeleteBuffers(1, &box_vbo_);
  glDeleteTextures(1, &color_tex_);
  glDeleteRenderbuffers(1, &depth_rb_);
  glDeleteFramebuffers(1, &fbo_);
  sphere_prog_ = line_prog_ = 0;
  quad_vao_ = box_vao_ = quad_vbo_ = instance_vbo_ = box_vbo_ = 0;
  color_tex_ = depth_rb_ = fbo_ = 0;
}

} // namespace rmcgui
