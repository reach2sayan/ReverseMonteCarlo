#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

// Eigen 5 moved the slicing placeholders out of the top-level Eigen namespace
// into Eigen::placeholders, so the Eigen 3.x spelling `Eigen::all` (used as the
// all-columns selector in coords(indices, Eigen::all) throughout the generators)
// no longer resolves — worse, it silently binds to the unrelated internal
// helper Eigen::internal::all(). Re-expose the placeholder under its historical
// name once, here, so every call site keeps working without per-site edits.
// (Guarded so it's a no-op on Eigen < 5, where Eigen::all already exists.)
#if EIGEN_VERSION_AT_LEAST(5, 0, 0)
namespace Eigen {
using placeholders::all;
}
#endif
#include <boost/leaf/result.hpp>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <span>
#include <string>
#include <thread>

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline
#endif
namespace RMC {

using index_t = std::size_t;
using real_t = double;
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

constexpr auto upper_triangle_pairs(auto &&N) {
  return std::views::iota(decltype(N){0}, N) |
         std::views::transform([N](auto i) {
           return std::views::iota(i + 1, N) |
                  std::views::transform(
                      [i](auto j) { return std::pair{i, j}; });
         }) |
         std::views::join;
}

// Returns the number of CPUs allocated to this process, in priority order:
//   1. SLURM_CPUS_PER_TASK  (SLURM scheduler)
//   2. PBS_NUM_PPN           (PBS/Torque scheduler)
//   3. LSB_DJOB_NUMPROC      (LSF scheduler)
//   4. std::thread::hardware_concurrency() (local fallback)
inline std::size_t allocated_cpus() noexcept {
  for (const char *var :
       {"SLURM_CPUS_PER_TASK", "PBS_NUM_PPN", "LSB_DJOB_NUMPROC"}) {
    if (const char *val = std::getenv(var); val && *val) {
      if (const int n = std::atoi(val); n > 0) {
        return static_cast<std::size_t>(n);
      }
    }
       }
  return std::max(1u, std::thread::hardware_concurrency());
}

} // namespace RMC
