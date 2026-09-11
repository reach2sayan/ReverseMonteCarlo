#pragma once
#include <boost/describe/enum.hpp>
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalCache.hpp>
#include <RMC/core/SpeciesIndex.hpp>
#include <boost/container/flat_map.hpp>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace RMC {

enum class DistanceScope : std::uint8_t { Inter, Intra };
BOOST_DESCRIBE_ENUM(DistanceScope, Inter, Intra)

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
  using RigidConstraintBase<DistanceConstraint<S>>::bc_;

public:
  void set_minimum_distance(const std::string &el1, const std::string &el2,
                            double d_min) {
    d_min_.insert_or_assign(PairElemKey{el1, el2}, d_min);
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
    if constexpr (S == DistanceScope::Inter) {
      return "InterMolecularDistanceConstraint";
    } else {
      return "IntraMolecularDistanceConstraint";
    }
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    if (!cache_.ready) {
      build_thresholds();
    }
    const std::size_t n = species_.size();
    return cache_.compute(
        static_cast<std::size_t>(coords.rows()),
        [&](std::size_t i, std::size_t j) -> std::optional<double> {
          return in_scope(i, j) ? thresholds_[species_.id[i] * n + species_.id[j]]
                                : std::nullopt;
        },
        [&](std::size_t i, std::size_t j) {
          if (this->absent(i) || this->absent(j)) {
            return std::numeric_limits<double>::infinity(); // pair gone
          }
          const vec3_t d = coords.row(j).transpose() - coords.row(i).transpose();
          return (bc_ ? bc_->min_image(d) : d).squaredNorm();
        },
        moved);
  }

private:
  // Dense species × species minimum-distance table (nullopt = unconstrained).
  void build_thresholds() const {
    species_ = SpeciesIndex(elements_);
    const std::size_t n = species_.size();
    thresholds_.assign(n * n, std::nullopt);
    for (const auto &[key, d] : d_min_) {
      if (const auto a = species_.id_of(key.a), b = species_.id_of(key.b);
          a && b) {
        thresholds_[*a * n + *b] = d;
        thresholds_[*b * n + *a] = d;
      }
    }
  }

  [[nodiscard]] FORCE_INLINE bool in_scope(std::size_t i,
                                           std::size_t j) const noexcept {
    if (mol_ids_.empty()) {
      return true;
    }
    if constexpr (S == DistanceScope::Inter) {
      return mol_ids_[i] != mol_ids_[j];
    } else {
      return mol_ids_[i] == mol_ids_[j];
    }
  }

  boost::container::flat_map<PairElemKey, double> d_min_;
  std::span<const std::string> elements_;
  std::span<const std::size_t> mol_ids_;
  mutable SpeciesIndex species_;
  mutable std::vector<std::optional<double>> thresholds_;
  mutable PairCache cache_;
};

using InterMolecularDistanceConstraint =
    DistanceConstraint<DistanceScope::Inter>;
using IntraMolecularDistanceConstraint =
    DistanceConstraint<DistanceScope::Intra>;

} // namespace RMC
