#pragma once
#include <fullrmc/constraints/Constraint.hpp>
#include <boost/container/flat_map.hpp>
#include <string>
#include <vector>
#include <cmath>

namespace fullrmc {

// Minimum-distance constraint between atom-type pairs.
// std_err = Σ max(0, d_min - d_ij) for all pairs (i,j) within scope.
//
// Two specialisations: intra-molecular (same molecule_id) and
// inter-molecular (different molecule_id). Both share the same base.
class DistanceConstraintBase : public ConstraintBase<DistanceConstraintBase> {
public:
    // Key: sorted pair of element symbols, e.g. {"H","O"}.
    using ElemPair = std::pair<std::string, std::string>;

    void set_minimum_distance(const std::string& el1,
                               const std::string& el2,
                               real_t d_min) {
        auto key = el1 < el2 ? ElemPair{el1, el2} : ElemPair{el2, el1};
        d_min_[key] = d_min;
    }

    void set_structure(const std::vector<std::string>* elements,
                        const std::vector<index_t>*     molecule_ids) noexcept {
        elements_    = elements;
        mol_ids_     = molecule_ids;
    }

    [[nodiscard]] std::string name() const override { return "DistanceConstraint"; }

    [[nodiscard]] real_t compute_error(const coords_t& coords,
                                        std::span<const index_t> /*moved*/) const {
        real_t err = 0.0;
        const std::size_t N = static_cast<std::size_t>(coords.rows());
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = i + 1; j < N; ++j)
                if (in_scope(static_cast<index_t>(i), static_cast<index_t>(j))) {
                    auto key = make_key(static_cast<index_t>(i),
                                        static_cast<index_t>(j));
                    auto it = d_min_.find(key);
                    if (it == d_min_.end()) continue;
                    real_t d = distance(coords, static_cast<index_t>(i),
                                         static_cast<index_t>(j));
                    if (d < it->second) err += (it->second - d);
                }
        return err;
    }

protected:
    virtual bool in_scope(index_t i, index_t j) const noexcept = 0;

    [[nodiscard]] ElemPair make_key(index_t i, index_t j) const {
        const std::string& a = (*elements_)[static_cast<std::size_t>(i)];
        const std::string& b = (*elements_)[static_cast<std::size_t>(j)];
        return a < b ? ElemPair{a, b} : ElemPair{b, a};
    }

    boost::container::flat_map<ElemPair, real_t> d_min_;
    const std::vector<std::string>* elements_  = nullptr;
    const std::vector<index_t>*     mol_ids_   = nullptr;
};

class InterMolecularDistanceConstraint : public DistanceConstraintBase {
    [[nodiscard]] std::string name() const override { return "InterMolecularDistanceConstraint"; }
protected:
    bool in_scope(index_t i, index_t j) const noexcept override {
        if (!mol_ids_) return true;
        return (*mol_ids_)[i] != (*mol_ids_)[j];
    }
};

class IntraMolecularDistanceConstraint : public DistanceConstraintBase {
    [[nodiscard]] std::string name() const override { return "IntraMolecularDistanceConstraint"; }
protected:
    bool in_scope(index_t i, index_t j) const noexcept override {
        if (!mol_ids_) return true;
        return (*mol_ids_)[i] == (*mol_ids_)[j];
    }
};

} // namespace fullrmc
