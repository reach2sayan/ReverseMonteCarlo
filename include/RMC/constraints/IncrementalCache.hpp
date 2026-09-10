#pragma once
#include <Eigen/Core>
#include <RMC/core/Types.hpp>
#include <cmath>
#include <cstdint>
#include <ranges>
#include <span>
#include <unordered_map>
#include <vector>

namespace RMC {

// Type 1: ItemCache — incremental cache for constraints over a fixed list of
// items (bonds, angles, …). On a move, only items including a moved atom are
// recomputed. Item is the constraint's item struct (Bond, Triplet, …).
//   compute(items_, atoms_of, err_of, coords, moved)
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
      for (auto &&[i, item] :
           std::views::zip(items, errs) | std::views::enumerate) {
        auto &[itemi, err] = item;
        err = err_of(coords, itemi);
        total += err;
        for (auto a : atoms_of(itemi)) {
          atom_map[a].push_back(i);
        }
      }
      ready = true;
      return total;
    }
    for (auto atom : moved) {
      auto it = atom_map.find(atom);
      if (it != atom_map.end()) {
        for (auto idx : it->second) {
          const double ne = err_of(coords, items[idx]);
          total += ne - errs[idx];
          errs[idx] = ne;
        }
      }
    }
    return total;
  }
};

// Type 2: PairCache — incremental cache for O(N²) pair-distance constraints.
// fwd[i] = {(j, threshold, contrib) | j>i, eligible}, bwd[j] = {(i, pos_in_fwd[i]) | i<j}.
// A single-atom move at k touches fwd[k] (O(N-k)) and bwd[k] (O(k)), so each step is O(N).
//   compute(N, pair_threshold, dist, coords, moved)
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
  // Membership flags for the multi-atom backward pass (1 = atom is in `moved`).
  mutable std::vector<std::uint8_t> in_moved;

  constexpr void invalidate() noexcept { ready = false; }

  // PairFn : (i, j) → std::optional<double>  (threshold; nullopt = skip pair)
  // DistFn : (i, j) → double  (SQUARED distance — compared against threshold²;
  //          sqrt is taken only for the few pairs in violation)
  template <typename PairFn, typename DistFn>
  double build(std::size_t N, PairFn pair_threshold, DistFn dist) const {
    fwd.assign(N, {});
    bwd.assign(N, {});
    atom_contrib.setZero(static_cast<Eigen::Index>(N));
    auto indices = std::views::iota(std::size_t{0}, N);
    auto pairwise = std::views::cartesian_product(indices, indices) |
                    std::views::filter([](auto p) {
                      auto [a, b] = p;
                      return a < b;
                    }) |
                    std::views::transform([&](auto p) {
                      auto [a, b] = p;
                      return std::tuple{a, b, pair_threshold(a, b)};
                    }) |
                    std::views::filter([](const auto &t) {
                      return std::get<2>(t).has_value();
                    });
    for (auto [i, j, thresh] : pairwise) {
      const double d2 = dist(i, j);
      const double c =
          (d2 < *thresh * *thresh) ? (*thresh - std::sqrt(d2)) : 0.0;
      const std::size_t pos = fwd[i].size();
      fwd[i].push_back({j, *thresh, c});
      bwd[j].push_back({i, pos});
      atom_contrib(static_cast<Eigen::Index>(i)) += c;
    }
    ready = true;
    return atom_contrib.sum();
  }

  template <typename DistFn>
  double update(std::span<const std::size_t> moved,
                DistFn dist) const noexcept {
    // Multi-atom moves skip back-refs whose i is also in `moved` (their forward
    // pass already handles that pair), via the O(1) in_moved flag lookup.
    const bool multi = moved.size() > 1;
    if (multi) {
      if (in_moved.size() < fwd.size()) {
        in_moved.assign(fwd.size(), 0);
      }
      for (std::size_t m : moved) {
        in_moved[m] = 1;
      }
    }
    for (std::size_t k : moved) {
      // 1. Recompute forward pairs (k, j>k). `dist` is SQUARED; sqrt only on violation.
      double c_k = 0.0;
      for (auto &p : fwd[k]) {
        const double d2 = dist(k, p.j);
        const double nc =
            (d2 < p.threshold * p.threshold) ? (p.threshold - std::sqrt(d2))
                                             : 0.0;
        p.contrib = nc;
        c_k += nc;
      }
      atom_contrib(static_cast<Eigen::Index>(k)) = c_k;

      // 2. Update backward pairs (i<k); skip i if also in moved (its forward
      //    pass handles pair (i,k)).
      auto update_bwd = [&](const BackRef &br) {
        auto &p = fwd[br.i][br.pos];
        const double d2 = dist(br.i, k);
        const double nc =
            (d2 < p.threshold * p.threshold) ? (p.threshold - std::sqrt(d2))
                                             : 0.0;
        atom_contrib(static_cast<Eigen::Index>(br.i)) += nc - p.contrib;
        p.contrib = nc;
      };
      if (!multi) {
        std::ranges::for_each(bwd[k], [&](const auto &br) { update_bwd(br); });
      } else {
        for (const auto &br : bwd[k]) {
          if (!in_moved[br.i]) {
            update_bwd(br);
          }
        }
      }
    }
    if (multi) {
      for (std::size_t m : moved) {
        in_moved[m] = 0;
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
