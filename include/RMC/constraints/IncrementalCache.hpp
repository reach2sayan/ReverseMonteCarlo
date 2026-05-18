#pragma once
#include <Eigen/Core>
#include <RMC/core/Types.hpp>
#include <ranges>
#include <span>
#include <unordered_map>
#include <vector>

namespace RMC {

// ── Type 1: ItemCache ────────────────────────────────────────────────────────
// Incremental cache for constraints defined over a fixed list of items (bonds,
// angles, dihedrals, …). Each item touches a small set of atoms. On a move,
// only items that include a moved atom are recomputed; all others retain their
// cached error contribution.
//
// Usage in compute_error():
//   return cache_.compute(items_, atoms_of_lambda, err_of_lambda, coords,
//   moved);
//
// Template parameter Item is the constraint's item struct (Bond, Triplet,
// Quad…).
template <typename Item> struct ItemCache {
  mutable bool ready{false};
  mutable double total{0.0};
  mutable std::vector<double> errs;
  mutable std::unordered_map<std::size_t, std::vector<std::size_t>> atom_map;

  constexpr void invalidate() noexcept { ready = false; }

  // AtomsOf : Item → iterable of std::size_t  (e.g. std::array<std::size_t,2>)
  // ErrOf   : (const coords_t&, const Item&) → double
  template <typename AtomsOf, typename ErrOf>
  double compute(std::span<const Item> items, AtomsOf atoms_of, ErrOf err_of,
                 const coords_t &coords,
                 std::span<const std::size_t> moved) const {
    if (!ready || moved.empty()) {
      total = 0.0;
      atom_map.clear();
      errs.assign(items.size(), 0.0);
      for (std::size_t i = 0; i < items.size(); ++i) {
        errs[i] = err_of(coords, items[i]);
        total += errs[i];
        for (auto a : atoms_of(items[i])) {
          atom_map[a].push_back(i);
        }
      }
      ready = true;
      return total;
    }
    for (auto atom : moved) {
      auto it = atom_map.find(atom);
      if (it == atom_map.end()) {
        continue;
      }
      for (auto idx : it->second) {
        const double ne = err_of(coords, items[idx]);
        total += ne - errs[idx];
        errs[idx] = ne;
      }
    }
    return total;
  }
};

// ── Type 2: PairCache ────────────────────────────────────────────────────────
// Incremental cache for O(N²) pair-distance constraints. Stores forward refs
// fwd[i] = {(j, threshold, contrib) | j > i, pair eligible} and backward refs
// bwd[j] = {(i, pos_in_fwd[i]) | i < j}. A single-atom move at k touches only
// fwd[k] (O(N-k) pairs) and bwd[k] (O(k) back-refs), keeping each step O(N).
//
// Usage in compute_error():
//   return cache_.compute(N, pair_threshold_lambda, dist_lambda, coords,
//   moved);
struct PairCache {
  struct FwdPair {
    std::size_t j;
    double threshold;
    mutable double contrib{0.0};
  };
  struct BackRef {
    std::size_t i;
    std::size_t pos; // index into fwd[i]
  };

  mutable bool ready{false};
  mutable std::vector<std::vector<FwdPair>> fwd;
  mutable std::vector<std::vector<BackRef>> bwd;
  mutable Eigen::VectorXd atom_contrib;

  void invalidate() noexcept { ready = false; }

  // PairFn : (i, j) → std::optional<double>  (threshold; nullopt = skip pair)
  // DistFn : (i, j) → double
  template <typename PairFn, typename DistFn>
  double build(std::size_t N, PairFn pair_threshold, DistFn dist) const {
    fwd.assign(N, {});
    bwd.assign(N, {});
    atom_contrib.setZero(static_cast<Eigen::Index>(N));
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = i + 1; j < N; ++j) {
        auto thresh = pair_threshold(i, j);
        if (!thresh)
          continue;
        const double d = dist(i, j);
        const double c = (d < *thresh) ? (*thresh - d) : 0.0;
        const std::size_t pos = fwd[i].size();
        fwd[i].push_back({j, *thresh, c});
        bwd[j].push_back({i, pos});
        atom_contrib(static_cast<Eigen::Index>(i)) += c;
      }
    }
    ready = true;
    return atom_contrib.sum();
  }

  template <typename DistFn>
  double update(std::span<const std::size_t> moved, DistFn dist) const {
    for (std::size_t k : moved) {
      // 1. Recompute all forward pairs (k, j) with j > k.
      double c_k = 0.0;
      for (auto &p : fwd[k]) {
        const double d = dist(k, p.j);
        const double nc = (d < p.threshold) ? (p.threshold - d) : 0.0;
        p.contrib = nc;
        c_k += nc;
      }
      atom_contrib(static_cast<Eigen::Index>(k)) = c_k;

      // 2. Update backward pairs (i, k) with i < k.
      //    Skip i if it is also in moved — its forward pass handles pair (i,k).
      for (const auto &[i, pos] : bwd[k] | std::views::filter([&](const auto& br) {
        return std::ranges::find(moved, br.i) == moved.end();
      })) {
        auto &p = fwd[i][pos];
        const double d = dist(i, k);
        const double nc = (d < p.threshold) ? (p.threshold - d) : 0.0;
        atom_contrib(static_cast<Eigen::Index>(i)) += nc - p.contrib;
        p.contrib = nc;
      }
    }
    return atom_contrib.sum();
  }

  template <typename PairFn, typename DistFn>
  constexpr double compute(std::size_t N, PairFn pair_threshold, DistFn dist,
                 std::span<const std::size_t> moved) const {
    return (!ready || moved.empty()) ? build(N, pair_threshold, dist)
                                     : update(moved, dist);
  }
};

} // namespace RMC
