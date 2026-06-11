#include "Camera.hpp"

#include <Eigen/Geometry> // Vector3f::cross() is defined here, not in Core

#include <algorithm>
#include <cmath>

namespace rmcgui {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kPitchLimit = kPi * 0.49f; // keep off the poles
} // namespace

Eigen::Vector3f Camera::forward() const {
  // Spherical → Cartesian: the unit vector from target toward the eye.
  const float cp = std::cos(pitch_);
  return {cp * std::sin(yaw_), std::sin(pitch_), cp * std::cos(yaw_)};
}

Eigen::Vector3f Camera::eye() const { return target_ + forward() * distance_; }

void Camera::orbit(float dx, float dy) {
  yaw_ -= dx * orbit_scale_;
  pitch_ = std::clamp(pitch_ + dy * orbit_scale_, -kPitchLimit, kPitchLimit);
}

void Camera::zoom(float wheel) {
  // Multiplicative so each notch is a constant fraction.
  distance_ *= std::pow(0.9f, wheel);
  distance_ = std::max(distance_, 1e-3f);
}

void Camera::pan(float dx, float dy) {
  const Eigen::Vector3f f = forward();
  const Eigen::Vector3f world_up(0.0f, 1.0f, 0.0f);
  Eigen::Vector3f right = world_up.cross(f);
  if (right.norm() < 1e-5f) {
    right = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
  }
  right.normalize();
  const Eigen::Vector3f up = f.cross(right).normalized();
  const float s = pan_scale_ * distance_;
  target_ += (-dx * s) * right + (dy * s) * up;
}

void Camera::frame(const Eigen::Vector3f &center, float radius) {
  target_ = center;
  const float r = std::max(radius, 1e-3f);
  // Distance so the sphere fits the vertical FOV, with a little margin.
  distance_ = 1.4f * r / std::sin(fov_y_ * 0.5f);
}

Eigen::Matrix4f Camera::view() const {
  const Eigen::Vector3f e = eye();
  const Eigen::Vector3f f = (target_ - e).normalized(); // look direction
  const Eigen::Vector3f world_up(0.0f, 1.0f, 0.0f);
  Eigen::Vector3f s = f.cross(world_up);
  if (s.norm() < 1e-5f) {
    s = f.cross(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
  }
  s.normalize();
  const Eigen::Vector3f u = s.cross(f);

  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m(0, 0) = s.x();  m(0, 1) = s.y();  m(0, 2) = s.z();
  m(1, 0) = u.x();  m(1, 1) = u.y();  m(1, 2) = u.z();
  m(2, 0) = -f.x(); m(2, 1) = -f.y(); m(2, 2) = -f.z();
  m(0, 3) = -s.dot(e);
  m(1, 3) = -u.dot(e);
  m(2, 3) = f.dot(e);
  return m;
}

Eigen::Matrix4f Camera::proj(float aspect) const {
  const float a = std::max(aspect, 1e-3f);
  const float t = 1.0f / std::tan(fov_y_ * 0.5f);
  Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
  m(0, 0) = t / a;
  m(1, 1) = t;
  m(2, 2) = (far_ + near_) / (near_ - far_);
  m(2, 3) = (2.0f * far_ * near_) / (near_ - far_);
  m(3, 2) = -1.0f;
  return m;
}

} // namespace rmcgui
