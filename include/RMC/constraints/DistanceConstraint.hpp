#pragma once
#include <boost/container/flat_map.hpp>
#include <cmath>
#include <RMC/constraints/Constraint.hpp>
#include <string>
#include <vector>

namespace RMC {

// Minimum-distance constraint between atom-type pairs.
// std_err = Σ max(0, d_min - d_ij) for all pairs (i,j) within scope.
//
// Two specialisations: intra-molecular (same molecule_id) and
// inter-molecular (different molecule_id). Both share the same base.
class DistanceConstraintBase : public ConstraintBase<DistanceConstraintBase> {
public:
  // Key: sorted pair of element symbols, e.g. {"H","O"}.
  using ElemPair = std::pair<std::string, std::string>;
  constexpr void set_minimum_distance(const std::string &el1,
                                      const std::string &el2, double d_min) {
    auto key = el1 < el2 ? ElemPair{el1, el2} : ElemPair{el2, el1};
    d_min_[key] = d_min;
  }

  constexpr void
  set_structure(const std::vector<std::string> *elements,
                const std::vector<std::size_t> *molecule_ids) noexcept {
    elements_ = elements;
    mol_ids_ = molecule_ids;
  }

  [[nodiscard]] std::string name() const {
    return "DistanceConstraint";
  }

  [[nodiscard]] double
  compute_error(const coords_t &coords,
                std::span<const std::size_t> /*moved*/) const {
    double err = 0.0;
    const std::size_t N = static_cast<std::size_t>(coords.rows());
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = i + 1; j < N; ++j)
        if (in_scope(static_cast<std::size_t>(i),
                     static_cast<std::size_t>(j))) {
          auto key = make_key(static_cast<std::size_t>(i),
                              static_cast<std::size_t>(j));
          auto it = d_min_.find(key);
          if (it == d_min_.end()) {
            continue;
          }
          double d = distance(coords, static_cast<std::size_t>(i),
                              static_cast<std::size_t>(j));
          if (d < it->second) {
            err += (it->second - d);
          }
        }
    return err;
  }

protected:
  virtual bool in_scope(std::size_t i, std::size_t j) const noexcept = 0;

  [[nodiscard]] ElemPair make_key(std::size_t i, std::size_t j) const {
    const std::string &a = (*elements_)[static_cast<std::size_t>(i)];
    const std::string &b = (*elements_)[static_cast<std::size_t>(j)];
    return a < b ? ElemPair{a, b} : ElemPair{b, a};
  }

  boost::container::flat_map<ElemPair, double> d_min_;
  const std::vector<std::string> *elements_ = nullptr;
  const std::vector<std::size_t> *mol_ids_ = nullptr;
};

class InterMolecularDistanceConstraint : public DistanceConstraintBase {
public:
  [[nodiscard]] std::string name() const {
    return "InterMolecularDistanceConstraint";
  }

protected:
  bool in_scope(std::size_t i, std::size_t j) const noexcept override {
    return (!mol_ids_) ? true : (*mol_ids_)[i] != (*mol_ids_)[j];
  }
};

class IntraMolecularDistanceConstraint : public DistanceConstraintBase {
public:
  [[nodiscard]] std::string name() const {
    return "IntraMolecularDistanceConstraint";
  }

protected:
  constexpr bool in_scope(std::size_t i,
                          std::size_t j) const noexcept override {
    return (!mol_ids_) ? true : (*mol_ids_)[i] == (*mol_ids_)[j];
  }
};

} // namespace RMC
