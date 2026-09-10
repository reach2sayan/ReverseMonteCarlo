#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/RngGenerator.hpp>
#include <RMC/core/Types.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_erasure/any.hpp>
#include <boost/type_erasure/builtin.hpp>
#include <boost/type_erasure/member.hpp>
#include <optional>
#include <span>

namespace RMC {

namespace detail {
BOOST_TYPE_ERASURE_MEMBER(generate)
BOOST_TYPE_ERASURE_MEMBER(rejection_override)
BOOST_TYPE_ERASURE_MEMBER(modifies_species)
} // namespace detail

template <typename T>
concept CMoveGenerator = requires(T &gen, coords_t &coords,
                                  std::span<const std::size_t> indices) {
  gen.generate(coords, indices);
};

// Base of every generator: defaults for the optional hooks.
template <typename Derived> struct MoveGeneratorBase {
  // A generator that runs its own accept/reject (HMC) returns the verdict here;
  // Engine::settle() then defers to it.
  [[nodiscard]] static constexpr std::optional<bool>
  rejection_override() noexcept {
    return std::nullopt;
  }
  // A generator that mutates species (not just coordinates) returns true so the
  // engine snapshots and restores them around the move.
  [[nodiscard]] static constexpr bool modifies_species() noexcept {
    return false;
  }
};

// A move generator held by value (Boost.TypeErasure): anything deriving
// MoveGeneratorBase with generate(coords, indices) converts to it.
using MoveGenerator = boost::type_erasure::any<boost::mpl::vector<
    boost::type_erasure::copy_constructible<>, boost::type_erasure::relaxed,
    detail::has_generate<void(coords_t &, std::span<const std::size_t>)>,
    detail::has_rejection_override<std::optional<bool>(),
                                   const boost::type_erasure::_self>,
    detail::has_modifies_species<bool(), const boost::type_erasure::_self>>>;

enum class SymmetryAxis { X, Y, Z };
[[nodiscard]] inline vec3_t unit(SymmetryAxis a) {
  return vec3_t::Unit(static_cast<Eigen::Index>(a));
}

// --- pieces shared by the concrete generators ------------------------------

FORCE_INLINE vec3_t centroid(const coords_t &coords,
                             std::span<const std::size_t> indices) noexcept {
  return coords(indices, Eigen::all).colwise().mean();
}

// A magnitude drawn uniformly from [lo, hi] (lo when the range is empty).
struct Amplitude {
  double lo{0.0};
  double hi{0.2};
  [[nodiscard]] double draw(Rng &rng) const {
    return lo < hi ? rng.uniform(lo, hi) : lo;
  }
  [[nodiscard]] double signed_draw(Rng &rng) const {
    const double a = draw(rng);
    return rng.uniform() < 0.5 ? -a : a;
  }
};

// Isotropic unit vector: a normalised Gaussian triple.
[[nodiscard]] inline vec3_t random_unit_vector(Rng &rng) {
  const double x = rng.normal(), y = rng.normal(), z = rng.normal();
  const vec3_t v(x, y, z);
  const double n = v.norm();
  return n < 1e-12 ? vec3_t(vec3_t::UnitX()) : vec3_t(v / n);
}

inline void translate(coords_t &coords, std::span<const std::size_t> indices,
                      const vec3_t &delta) {
  coords(indices, Eigen::all).rowwise() += delta.transpose();
}

// Rigid rotation of the rows `indices` by R about `pivot`.
inline void rotate_about(coords_t &coords, std::span<const std::size_t> indices,
                         const mat3_t &R, const vec3_t &pivot) {
  auto rows = coords(indices, Eigen::all);
  rows = ((rows.rowwise() - pivot.transpose()) * R.transpose()).rowwise() +
         pivot.transpose();
}

// Rotation by `angle` about the unit `axis` through the group centroid.
inline void rotate_group(coords_t &coords, std::span<const std::size_t> indices,
                         double angle, const vec3_t &axis) {
  rotate_about(coords, indices,
               Eigen::AngleAxisd(angle, axis).toRotationMatrix(),
               centroid(coords, indices));
}

} // namespace RMC
