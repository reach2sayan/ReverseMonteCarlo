#pragma once
#include <Eigen/Core>
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalCache.hpp>
#include <boost/container/flat_map.hpp>
#include <string>

namespace RMC {

enum class DistanceScope : std::uint8_t { Inter, Intra };

// Minimum-distance constraint between atom-type pairs.
// std_err = Σ max(0, d_min - d_ij) for all in-scope pairs (i,j).
//
// The scope policy (inter- vs intra-molecular) is a compile-time template
// parameter resolved via if constexpr — no virtual dispatch, no vptr.
//
// Incremental: PairCache stores per-pair contributions with forward and
// backward refs, so a single-atom move at k recomputes only the O(N) pairs
// involving k, not O(N²) pairs total.
template <DistanceScope S>
class DistanceConstraint : public RigidConstraintBase<DistanceConstraint<S>> {
  struct ElemPair {
    std::string key1;
    std::string key2;
    constexpr ElemPair(std::string a, std::string b)
        : key1(std::move(a)), key2(std::move(b)) {
      if (key1 > key2)
        std::swap(key1, key2);
    }
    constexpr auto operator<=>(const ElemPair &) const = default;
  };

public:
  void set_minimum_distance(const std::string &el1, const std::string &el2,
                            double d_min) {
    d_min_[ElemPair{el1, el2}] = d_min;
    cache_.invalidate();
  }

  constexpr void
  set_structure(std::span<const std::string> elements,
                std::span<const std::size_t> molecule_ids) noexcept {
    elements_ = elements;
    mol_ids_ = molecule_ids;
    cache_.invalidate();
  }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    if constexpr (S == DistanceScope::Inter)
      return "InterMolecularDistanceConstraint";
    else
      return "IntraMolecularDistanceConstraint";
  }

  [[nodiscard]] constexpr double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    const std::size_t N = static_cast<std::size_t>(coords.rows());
    return cache_.compute(
        N,
        [&](std::size_t i, std::size_t j) -> std::optional<double> {
          if (!in_scope(i, j)) {
            return std::nullopt;
          }
          auto it = d_min_.find(make_key(i, j));
          if (it == d_min_.end()) {
            return std::nullopt;
          }
          return it->second;
        },
        [&](std::size_t i, std::size_t j) {
          return this->distance(coords, i, j);
        },
        moved);
  }

private:
  [[nodiscard]] constexpr FORCE_INLINE bool
  in_scope(std::size_t i, std::size_t j) const noexcept {
    if (mol_ids_.empty()) {
      return true;
    }
    if constexpr (S == DistanceScope::Inter) {
      return mol_ids_[i] != mol_ids_[j];
    } else {
      return mol_ids_[i] == mol_ids_[j];
    }
  }

  [[nodiscard]] constexpr ElemPair make_key(std::size_t i,
                                            std::size_t j) const {
    return {elements_[i], elements_[j]};
  }

  boost::container::flat_map<ElemPair, double> d_min_;
  std::span<const std::string> elements_;
  std::span<const std::size_t> mol_ids_;

  mutable PairCache cache_;
};

using InterMolecularDistanceConstraint =
    DistanceConstraint<DistanceScope::Inter>;
using IntraMolecularDistanceConstraint =
    DistanceConstraint<DistanceScope::Intra>;

} // namespace RMC
