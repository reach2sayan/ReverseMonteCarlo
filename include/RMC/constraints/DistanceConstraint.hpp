#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <boost/container/flat_map.hpp>
#include <cmath>
#include <ranges>
#include <string>
#include <vector>

namespace RMC {

enum class DistanceScope { Inter, Intra };

// Minimum-distance constraint between atom-type pairs.
// std_err = Σ max(0, d_min - d_ij) for all in-scope pairs (i,j).
//
// The scope policy (inter- vs intra-molecular) is a compile-time template
// parameter resolved via if constexpr — no virtual dispatch, no vptr.
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
  }

  constexpr void
  set_structure(const std::vector<std::string> *elements,
                const std::vector<std::size_t> *molecule_ids) noexcept {
    elements_ = elements;
    mol_ids_ = molecule_ids;
  }

  [[nodiscard]] std::string name() const {
    if constexpr (S == DistanceScope::Inter) {
      return "InterMolecularDistanceConstraint";
    } else {
      return "IntraMolecularDistanceConstraint";
    }
  }

  [[nodiscard]] double
  compute_error(const coords_t &coords,
                std::span<const std::size_t> /*moved*/) const {
    double err = 0.0;
    const std::size_t N = static_cast<std::size_t>(coords.rows());

    auto upper_pairs =
        std::views::iota(std::size_t{0}, N) |
        std::views::transform([N](std::size_t i) {
          return std::views::iota(i + 1, N) |
                 std::views::transform(
                     [i](std::size_t j) { return std::pair{i, j}; });
        }) |
        std::views::join |
        std::views::filter([&](auto p) { return in_scope(p.first, p.second); });

    for (auto [i, j] : upper_pairs) {
      if (auto it = d_min_.find(make_key(i, j)); it != d_min_.end()) {
        double d = this->distance(coords, i, j);
        if (d < it->second)
          err += it->second - d;
      }
    }
    return err;
  }

private:
  [[nodiscard]] bool in_scope(std::size_t i, std::size_t j) const noexcept {
    if (!mol_ids_) {
      return true;
    }
    if constexpr (S == DistanceScope::Inter) {
      return (*mol_ids_)[i] != (*mol_ids_)[j];
    } else {
      return (*mol_ids_)[i] == (*mol_ids_)[j];
    }
  }

  [[nodiscard]] ElemPair make_key(std::size_t i, std::size_t j) const {
    return {elements_->at(i), elements_->at(j)};
  }

  boost::container::flat_map<ElemPair, double> d_min_;
  const std::vector<std::string> *elements_ = nullptr;
  const std::vector<std::size_t> *mol_ids_ = nullptr;
};

using InterMolecularDistanceConstraint =
    DistanceConstraint<DistanceScope::Inter>;
using IntraMolecularDistanceConstraint =
    DistanceConstraint<DistanceScope::Intra>;

} // namespace RMC
