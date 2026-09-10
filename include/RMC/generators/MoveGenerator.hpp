#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/TypeErasure.hpp>
#include <RMC/core/Types.hpp>
#include <memory>
#include <optional>

namespace RMC {

class MoveGenerator; // forward declaration for the friend declaration below

namespace detail {
struct MoveGeneratorToken {
private:
  constexpr MoveGeneratorToken() = default;
  friend class ::RMC::MoveGenerator;
};
} // namespace detail

template <typename T>
concept CMoveGenerator =
    requires(T &gen, detail::MoveGeneratorToken tok, coords_t &coords,
             std::span<const std::size_t> indices) {
      gen.generate(tok, coords, indices);
    };

// Optional extension: generator implements its own accept/reject (e.g. HMC).
// If satisfied, MoveGenerator exposes rejection_override() so Engine::settle()
// can defer to the generator instead of using constraints_.should_reject().
template <typename T>
concept CMoveGeneratorWithRejectionOverride =
    CMoveGenerator<T> && requires(const T &gen) {
      { gen.rejection_override() } -> std::convertible_to<std::optional<bool>>;
    };

// Optional extension: generator mutates species (elements/atomic_numbers), not
// just coordinates. Engine will save/restore species snapshots around such
// moves.
template <typename T>
concept CMoveGeneratorWithSpeciesModification =
    CMoveGenerator<T> && requires(const T &gen) {
      { gen.modifies_species() } -> std::convertible_to<bool>;
    };

// rejection_override() / modifies_species() are optional; default to
// std::nullopt / false. They forward without the passkey token.
#define RMC_MOVEGENERATOR_METHODS                                              \
  ((0, void, generate,                                                         \
    (coords_t & coords, std::span<const std::size_t> indices), 2,              \
    (coords, indices), , , WITH_TOKEN))
#define RMC_MOVEGENERATOR_OPT_METHODS                                          \
  ((1, std::optional<bool>, rejection_override, (), 0, (), const, noexcept,    \
    CMoveGeneratorWithRejectionOverride,                                       \
    std::nullopt))((1, bool, modifies_species, (), 0, (), const, noexcept,     \
                    CMoveGeneratorWithSpeciesModification, false))
RMC_DEFINE_ERASED_TYPE_EXT(MoveGenerator, RMC_MOVEGENERATOR_METHODS,
                           RMC_MOVEGENERATOR_OPT_METHODS)
#undef RMC_MOVEGENERATOR_METHODS
#undef RMC_MOVEGENERATOR_OPT_METHODS

enum class SymmetryAxis { X, Y, Z };

template <typename Derived> struct MoveGeneratorBase {};
FORCE_INLINE vec3_t centroid(const coords_t &coords,
                             std::span<const std::size_t> indices) noexcept {
  return coords(indices, Eigen::all).colwise().mean();
}

} // namespace RMC
