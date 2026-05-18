#pragma once
#include <Eigen/Core>
#include <RMC/constraints/Constraint.hpp>
#include <boost/container/flat_map.hpp>
#include <cmath>
#include <ranges>
#include <string>
#include <vector>

namespace RMC {

enum class DistanceScope : std::uint8_t { Inter, Intra };

// Minimum-distance constraint between atom-type pairs.
// std_err = Σ max(0, d_min - d_ij) for all in-scope pairs (i,j).
//
// The scope policy (inter- vs intra-molecular) is a compile-time template
// parameter resolved via if constexpr — no virtual dispatch, no vptr.
//
// Incremental: atom_contrib_[i] caches the error contribution from all pairs
// (i,j) with j > i. On a single-atom move, only the O(N) pairs touching that
// atom are recomputed, reducing per-step cost from O(N²) to O(N).
template <DistanceScope S>
class DistanceConstraint : public ConstraintBase<DistanceConstraint<S>> {
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
    initialised_ = false;
  }

  constexpr void set_structure(std::span<const std::string> elements,
                               std::span<const std::size_t> molecule_ids) noexcept {
    elements_ = elements;
    mol_ids_ = molecule_ids;
    initialised_ = false;
  }

  [[nodiscard]] std::string name() const {
    if constexpr (S == DistanceScope::Inter)
      return "InterMolecularDistanceConstraint";
    else
      return "IntraMolecularDistanceConstraint";
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    const std::size_t N = static_cast<std::size_t>(coords.rows());
    if (!initialised_ || moved.empty())
      return full_recompute(coords, N);

    for (std::size_t i : moved) {
      // Recompute atom_contrib_[i]: pairs (i,j) with j > i
      double c_i = 0.0;
      for (std::size_t j = i + 1; j < N; ++j) {
        if (!in_scope(i, j))
          continue;
        if (auto it = d_min_.find(make_key(i, j)); it != d_min_.end()) {
          double d = this->distance(coords, i, j);
          if (d < it->second)
            c_i += it->second - d;
        }
      }
      // Also fix contributions atom_contrib_[j] for all j < i (pair j<i involves i)
      for (std::size_t j = 0; j < i; ++j) {
        if (!in_scope(j, i))
          continue;
        auto it = d_min_.find(make_key(j, i));
        if (it == d_min_.end())
          continue;
        // atom_contrib_[j] stores sum over k>j; pair (j,i) is one such term.
        // Recompute full j row:
        double new_j = 0.0;
        for (std::size_t k = j + 1; k < N; ++k) {
          if (!in_scope(j, k))
            continue;
          auto it2 = d_min_.find(make_key(j, k));
          if (it2 == d_min_.end())
            continue;
          double dk = this->distance(coords, j, k);
          if (dk < it2->second)
            new_j += it2->second - dk;
        }
        atom_contrib_(static_cast<Eigen::Index>(j)) = new_j;
      }
      atom_contrib_(static_cast<Eigen::Index>(i)) = c_i;
    }
    return atom_contrib_.sum();
  }

private:
  double full_recompute(const coords_t &coords, std::size_t N) const {
    atom_contrib_.setZero(static_cast<Eigen::Index>(N));
    for (std::size_t i = 0; i < N; ++i) {
      double c = 0.0;
      for (std::size_t j = i + 1; j < N; ++j) {
        if (!in_scope(i, j))
          continue;
        if (auto it = d_min_.find(make_key(i, j)); it != d_min_.end()) {
          double d = this->distance(coords, i, j);
          if (d < it->second)
            c += it->second - d;
        }
      }
      atom_contrib_(static_cast<Eigen::Index>(i)) = c;
    }
    initialised_ = true;
    return atom_contrib_.sum();
  }

  [[nodiscard]] bool in_scope(std::size_t i, std::size_t j) const noexcept {
    if (mol_ids_.empty())
      return true;
    if constexpr (S == DistanceScope::Inter)
      return mol_ids_[i] != mol_ids_[j];
    else
      return mol_ids_[i] == mol_ids_[j];
  }

  [[nodiscard]] ElemPair make_key(std::size_t i, std::size_t j) const {
    return {elements_[i], elements_[j]};
  }

  boost::container::flat_map<ElemPair, double> d_min_;
  std::span<const std::string> elements_;
  std::span<const std::size_t> mol_ids_;

  mutable bool initialised_{false};
  mutable Eigen::VectorXd atom_contrib_;
};

using InterMolecularDistanceConstraint =
    DistanceConstraint<DistanceScope::Inter>;
using IntraMolecularDistanceConstraint =
    DistanceConstraint<DistanceScope::Intra>;

} // namespace RMC
