#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/leaf/result.hpp>
#include <concepts>
#include <cstdint>
#include <span>
#include <string>

namespace fullrmc {

using index_t = std::size_t;
using real_t  = double;
using vec3_t = Eigen::Vector3d;
using mat3_t = Eigen::Matrix3d;
using coords_t = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using vec_t = Eigen::VectorXd;
using mat_t = Eigen::MatrixXd;
using ivec_t = Eigen::VectorXi;

template <typename T>
concept BoundaryConditionsConcept = requires(T bc, const vec3_t &v) {
  { bc.wrap(v) } -> std::convertible_to<vec3_t>;
  { bc.min_image(v) } -> std::convertible_to<vec3_t>;
  { bc.volume() } -> std::convertible_to<double>;
};


template <typename T>
concept GroupSelectorConcept = requires(T s, std::size_t n) {
  { s.select(n) } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept ConstraintConcept =
    requires(T c, const coords_t &coords, std::span<const int> moved) {
      { c.compute_before_move(coords, moved) } -> std::same_as<void>;
      { c.compute_after_move(coords, moved) } -> std::same_as<void>;
      { c.standard_error() } -> std::convertible_to<double>;
      { c.should_reject() } -> std::convertible_to<bool>;
      { c.accept() } -> std::same_as<void>;
      { c.reject() } -> std::same_as<void>;
    };

template <typename T> using Result = boost::leaf::result<T>;

} // namespace fullrmc
