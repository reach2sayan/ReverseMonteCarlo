#pragma once
#include <Eigen/Core>
#include <RMC/core/Types.hpp>
#include <algorithm>
#include <boost/container/small_vector.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <ranges>
#include <span>
#include <vector>

namespace RMC {

// ItemCache — incremental sum over a fixed list of items (bonds, angles, …).
// On a move, only items including a moved atom are recomputed.
//   compute(items, atoms_of, err_of, coords, moved)
template <typename Item> struct ItemCache {
  mutable bool ready{false};
  mutable double total{0.0};
  mutable std::vector<double> errs;
  // Items touching each atom.
  mutable std::vector<boost::container::small_vector<std::size_t, 4>> items_of;
  constexpr void invalidate() noexcept { ready = false; }

  // AtomsOf : Item → iterable of std::size_t  (e.g. std::array<std::size_t,2>)
  // ErrOf   : (const coords_t&, const Item&) → double
  template <typename AtomsOf, typename ErrOf>
  double compute(std::span<const Item> items, AtomsOf atoms_of, ErrOf err_of,
                 const coords_t &coords,
                 std::span<const std::size_t> moved) const {
    if (!ready || moved.empty()) {
      errs.resize(items.size());
      items_of.assign(static_cast<std::size_t>(coords.rows()), {});
      for (const auto [i, item] : items | std::views::enumerate) {
        errs[static_cast<std::size_t>(i)] = err_of(coords, item);
        for (const std::size_t a : atoms_of(item)) {
          items_of[a].push_back(static_cast<std::size_t>(i));
        }
      }
      total = std::ranges::fold_left(errs, 0.0, std::plus{});
      ready = true;
      return total;
    }
    for (const std::size_t atom : moved) {
      for (const std::size_t idx : items_of[atom]) {
        const double e = err_of(coords, items[idx]);
        total += e - errs[idx];
        errs[idx] = e;
      }
    }
    return total;
  }
};

// PairCache — incremental sum of pair overlaps max(0, t − d) for O(N²)
// minimum-distance constraints. fwd[i] = {(j, t, contrib) | j > i, constrained},
// bwd[j] = {(i, position in fwd[i]) | i < j}; a single-atom move at k touches
// fwd[k] (O(N-k)) and bwd[k] (O(k)), so each step is O(N).
//   compute(N, pair_threshold, dist, moved)
struct PairCache {
  struct FwdPair {
    std::size_t j;
    double threshold;
    double contrib{0.0};
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

  // Overlap of a pair at SQUARED distance d2 with minimum distance t; sqrt is
  // taken only for pairs in violation.
  static double overlap(double d2, double t) noexcept {
    return d2 < t * t ? t - std::sqrt(d2) : 0.0;
  }

  // PairFn : (i, j) → std::optional<double>  (threshold; nullopt = skip pair)
  // DistFn : (i, j) → double  (SQUARED distance)
  template <typename PairFn, typename DistFn>
  double build(std::size_t N, PairFn pair_threshold, DistFn dist) const {
    fwd.assign(N, {});
    bwd.assign(N, {});
    atom_contrib.setZero(static_cast<Eigen::Index>(N));
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = i + 1; j < N; ++j) {
        if (const auto t = pair_threshold(i, j)) {
          const double c = overlap(dist(i, j), *t);
          bwd[j].push_back({i, fwd[i].size()});
          fwd[i].push_back({j, *t, c});
          atom_contrib[static_cast<Eigen::Index>(i)] += c;
        }
      }
    }
    ready = true;
    return atom_contrib.sum();
  }

  template <typename DistFn>
  double update(std::span<const std::size_t> moved, DistFn dist) const noexcept {
    // Multi-atom moves skip back-refs whose i is also in `moved` (its forward
    // pass already handles that pair).
    const bool multi = moved.size() > 1;
    if (multi) {
      in_moved.resize(std::max(in_moved.size(), fwd.size()), 0);
      for (const std::size_t m : moved) {
        in_moved[m] = 1;
      }
    }
    for (const std::size_t k : moved) {
      double c_k = 0.0;
      for (auto &p : fwd[k]) {
        p.contrib = overlap(dist(k, p.j), p.threshold);
        c_k += p.contrib;
      }
      atom_contrib[static_cast<Eigen::Index>(k)] = c_k;
      for (const auto &br : bwd[k]) {
        if (multi && in_moved[br.i]) {
          continue;
        }
        auto &p = fwd[br.i][br.pos];
        const double nc = overlap(dist(br.i, k), p.threshold);
        atom_contrib[static_cast<Eigen::Index>(br.i)] += nc - p.contrib;
        p.contrib = nc;
      }
    }
    if (multi) {
      for (const std::size_t m : moved) {
        in_moved[m] = 0;
      }
    }
    return atom_contrib.sum();
  }

  template <typename PairFn, typename DistFn>
  double compute(std::size_t N, PairFn pair_threshold, DistFn dist,
                 std::span<const std::size_t> moved) const {
    return (!ready || moved.empty()) ? build(N, pair_threshold, dist)
                                     : update(moved, dist);
  }
};

} // namespace RMC
