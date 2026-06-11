#pragma once
#include <Eigen/Core>

namespace rmcgui {

// Simple orbit/zoom/pan camera. Angles in radians; distance is eye→target.
// Produces column-major float matrices ready for glUniformMatrix4fv (GL_FALSE).
class Camera {
public:
  // Mouse-driven controls (deltas in pixels / wheel notches).
  void orbit(float dx, float dy);
  void zoom(float wheel);
  void pan(float dx, float dy);

  // Frame a bounding sphere (centre, radius) so it fills the view.
  void frame(const Eigen::Vector3f &center, float radius);
  [[nodiscard]] Eigen::Matrix4f view() const;
  [[nodiscard]] Eigen::Matrix4f proj(float aspect) const;
  [[nodiscard]] const Eigen::Vector3f &target() const { return target_; }

private:
  [[nodiscard]] Eigen::Vector3f forward() const; // target→eye direction
  [[nodiscard]] Eigen::Vector3f eye() const;

  Eigen::Vector3f target_ = Eigen::Vector3f::Zero();
  float yaw_ = 0.6f;      // around +y
  float pitch_ = 0.4f;    // up/down, clamped away from poles
  float distance_ = 30.0f;

  float fov_y_ = std::numbers::pi_v<float> * 45.0/180.0;
  float near_ = 0.1f;
  float far_ = 5000.0f;
  float pan_scale_ = 0.0025f; // pixels → world, multiplied by distance
  float orbit_scale_ = 0.008f;
};

} // namespace rmcgui
