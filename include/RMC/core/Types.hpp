#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

// Eigen 5 moved slicing placeholders into Eigen::placeholders; re-expose
// Eigen::all under its historical name so call sites keep working.
#if EIGEN_VERSION_AT_LEAST(5, 0, 0)
namespace Eigen {
using placeholders::all;
}
#endif
#include <boost/leaf/result.hpp>
#include <boost/nowide/cstdlib.hpp>
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <thread>

// MSVC ignores the standard spelling (warning C5030) and implements the layout
// optimisation under its own vendor attribute instead.
#if defined(_MSC_VER)
#define RMC_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define RMC_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

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

// Read env var, nullopt if unset/empty.
inline std::optional<std::string> read_env(const char *name) {
  if (const char *val = boost::nowide::getenv(name); val && *val) {
    return std::string(val);
  }
  return std::nullopt;
}

// Returns the number of CPUs allocated to this process, in priority order:
//   1. SLURM_CPUS_PER_TASK  (SLURM scheduler)
//   2. PBS_NUM_PPN           (PBS/Torque scheduler)
//   3. LSB_DJOB_NUMPROC      (LSF scheduler)
//   4. std::thread::hardware_concurrency() (local fallback)
inline std::size_t allocated_cpus() noexcept {
  for (const char *var :
       {"SLURM_CPUS_PER_TASK", "PBS_NUM_PPN", "LSB_DJOB_NUMPROC"}) {
    if (const auto val = read_env(var)) {
      if (const int n = std::atoi(val->c_str()); n > 0) {
        return static_cast<std::size_t>(n);
      }
    }
  }
  return std::max(1u, std::thread::hardware_concurrency());
}

} // namespace RMC
