#pragma once
#include <cstdint>
#include <concepts>
#include <span>
#include <expected>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fullrmc {

// ---- Scalar and index aliases ----
using real_t  = double;
using index_t = std::int32_t;

// ---- Eigen aliases ----
using vec3_t    = Eigen::Vector3d;
using mat3_t    = Eigen::Matrix3d;
// N-atom coordinate matrix: row i = (x,y,z) of atom i
using coords_t  = Eigen::Matrix<real_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
using vec_t     = Eigen::VectorXd;
using mat_t     = Eigen::MatrixXd;
using ivec_t    = Eigen::VectorXi;

// ---- Core concepts ----

template<typename T>
concept BoundaryConditionsConcept = requires(T bc, const vec3_t& v) {
    { bc.wrap(v) }      -> std::convertible_to<vec3_t>;
    { bc.min_image(v) } -> std::convertible_to<vec3_t>;
    { bc.volume() }     -> std::convertible_to<real_t>;
};

template<typename T>
concept MoveGeneratorConcept = requires(T g, coords_t& c,
                                         std::span<const index_t> idx) {
    { g.generate(c, idx) } -> std::same_as<void>;
};

template<typename T>
concept GroupSelectorConcept = requires(T s, std::size_t n) {
    { s.select(n) } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept ConstraintConcept = requires(T c,
    const coords_t& coords, std::span<const index_t> moved) {
    { c.compute_before_move(coords, moved) } -> std::same_as<void>;
    { c.compute_after_move(coords, moved) }  -> std::same_as<void>;
    { c.standard_error() }                  -> std::convertible_to<real_t>;
    { c.should_reject() }                   -> std::convertible_to<bool>;
    { c.accept() }                          -> std::same_as<void>;
    { c.reject() }                          -> std::same_as<void>;
};

// ---- Error type for I/O ----
using Error = std::string;

template<typename T>
using Result = std::expected<T, Error>;

} // namespace fullrmc
