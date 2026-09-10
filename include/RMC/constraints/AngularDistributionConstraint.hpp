#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalHistogram.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/NeighborGrid.hpp>
#include <RMC/core/Types.hpp>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace RMC {

// Number of unordered leg pairs for n element types: n*(n+1)/2.
[[nodiscard]] constexpr FORCE_INLINE int adf_n_leg_pairs(int n_types) noexcept {
  return n_types * (n_types + 1) / 2;
}

// Packed index of the unordered leg pair (p, q) — caller passes p ≤ q — into
// the upper-triangular range [0, adf_n_leg_pairs), so the angle-triplet column
// order is deterministic.
[[nodiscard]] constexpr FORCE_INLINE int
adf_leg_pair_index(int p, int q, int n_types) noexcept {
  return p * (2 * n_types - p + 1) / 2 + (q - p);
}

// Total number of triplet columns: central element × unordered leg pair.
[[nodiscard]] constexpr FORCE_INLINE int adf_n_cols(int n_types) noexcept {
  return n_types * adf_n_leg_pairs(n_types);
}

// One triplet column in the canonical ADF layout: central species `central`,
// unordered leg pair (`leg_p` ≤ `leg_q`), at flat column `col` (central-id outer,
// leg-pair inner — matches adf_leg_pair_index).
struct AdfColumn {
  int central;
  int leg_p;
  int leg_q;
  int col;
};

// The canonical column layout for `n_types` species, in column order. Iterating
// this is the single source of truth for the (a, p, q) → col mapping that the
// histogram split, per-column scaling and target divisor all share.
[[nodiscard]] std::vector<AdfColumn> adf_columns(int n_types);

// Accumulate a raw bond-angle histogram into `hist` (length n_bins·adf_n_cols),
// zeroed first. Each central atom i and unordered neighbour pair (j,k) within
// `max_dis` adds 1 to the bin of angle j–i–k (over [0,π], n_bins) in column
// (central=elem_id[i], legs=elem_id[j],elem_id[k]). Full O(N·⟨n⟩²) recompute.
void accumulate_angle_histogram(vec_t &hist, const coords_t &coords,
                                const BoundaryConditions *bc,
                                const std::vector<uint8_t> &elem_id,
                                int n_types, double max_dis, int n_bins,
                                const AtomsCollector *collector = nullptr);

// Incremental companion to accumulate_angle_histogram: adds into `hist` (caller
// zeroes) every angle j–i–k in `coords` whose vertex set {apex i, leg j, leg k}
// intersects `moved`, once each, same layout. `grid` must be synced to `coords`.
// Used for: new_hist = saved − accumulate_moved_angles(old) + accumulate_moved_angles(new).
// Iterating by apex over {moved ∪ neighbours(moved)} keys each angle uniquely.
void accumulate_moved_angles(vec_t &hist, const coords_t &coords,
                             const BoundaryConditions *bc,
                             const std::vector<uint8_t> &elem_id, int n_types,
                             double max_dis, int n_bins,
                             std::span<const std::size_t> moved,
                             const NeighborGrid &grid,
                             const AtomsCollector *collector = nullptr);

// Soft constraint fitting the bond-angle distribution function (ADF), per element
// triplet, against a target (contributes to total chi²). Histogram is one flat
// vec_t, row-major angle_bin × triplet_column (stride n_cols_). Element ids are
// assigned by *sorted* symbol (as in analysis::compute_adf), so column order is
// independent of atom ordering.
class AngularDistributionConstraint
    : public SingularConstraintBase<AngularDistributionConstraint> {
public:
  using SingularConstraintBase<
      AngularDistributionConstraint>::set_boundary_conditions;

  // Experimental target. Column 0 = angle bin centres (rad, [0,π]); remaining
  // columns = per-triplet targets in canonical order (see adf_leg_pair_index).
  // Value-column count must equal adf_n_cols(n_types); validated in initialise().
  void set_experimental_data(const mat_t &data);

  constexpr void set_elements(std::span<const std::string> e) noexcept {
    elements_ = e;
  }
  constexpr void set_cutoff(double max_dis) noexcept { max_dis_ = max_dis; }
  constexpr void set_smoothing(int range) noexcept { smooth_range_ = range; }
  // true (default): scale-invariant residual (matches pair constraints).
  // false: unscaled L2 residual with full volume normalization.
  constexpr void set_scale_invariant(bool v) noexcept { scale_invariant_ = v; }

  // Full histogram + grid rebuild every `n` accepts to bound drift of the
  // `saved − D_old + D_new` update. 0 (default) disables.
  constexpr void set_resync_interval(unsigned n) noexcept { resync_every_ = n; }

  // Build species ids, per-column normalization, flatten/validate target.
  // Call after set_experimental_data and set_elements.
  void initialise();

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "AngularDistributionConstraint";
  }
  // Between PDF (1e6) and S(Q) (2e6); full ADF recompute is O(N·⟨n⟩²).
  [[nodiscard]] static constexpr double
  computation_cost(Constraint::Token) noexcept {
    return 1.5e6;
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

  // Public (non-token) multi-frame controls. set_active_frame_idx resets the
  // pending before-move delta (saved snapshot/grid belong to the previous frame).
  void set_n_frames(std::size_t n);
  void set_active_frame_idx(std::size_t k) noexcept { hist_.set_active_frame(k); }
  // rollback restores the active frame's histogram + grid; commit drops the
  // snapshot and drives the optional drift resync.
  void rollback_frame() noexcept;
  void commit_frame() noexcept;

  // Token-gated wrappers (CConstraint).
  void set_n_frames(Constraint::Token, std::size_t n) { set_n_frames(n); }
  void set_active_frame(Constraint::Token, std::size_t k) noexcept {
    set_active_frame_idx(k);
  }
  void initialise(Constraint::Token) { initialise(); }
  void reject(Constraint::Token tok) noexcept {
    SingularConstraintBase<AngularDistributionConstraint>::reject(tok);
    rollback_frame();
  }
  void accept(Constraint::Token tok) noexcept {
    SingularConstraintBase<AngularDistributionConstraint>::accept(tok);
    commit_frame();
  }

  [[nodiscard]] constexpr const vec_t &computed() const noexcept {
    return computed_;
  }
  [[nodiscard]] constexpr const vec_t &experimental_data() const noexcept {
    return exp_data_;
  }

private:
  void normalise_and_smooth(const coords_t &coords) const;

  // The neighbour grid backing the active frame (single-frame = one entry).
  [[nodiscard]] NeighborGrid &active_grid() const noexcept {
    return frame_grids_[hist_.active_frame_];
  }

  // ---- Configuration ----
  vec_t exp_theta_;     // angle bin centres (col 0 of the data file)
  mat_t exp_values_;    // staged value columns, flattened in initialise()
  vec_t exp_data_;      // flattened target, length hist_len_
  double max_dis_{3.4}; // bond cutoff (Å)
  int n_bins_{0};       // angle bins over [0, π]
  int smooth_range_{2}; // boxcar half-width (0 disables); applied in 2 passes
  bool scale_invariant_{true};
  std::span<const std::string> elements_;

  // ---- Derived (initialise) ----
  bool initialised_{false};
  int n_types_{0}, n_leg_pairs_{0}, n_cols_{0}, hist_len_{0};
  std::vector<uint8_t> elem_id_;     // per-atom sorted species id
  std::vector<std::string> species_; // sorted distinct symbols
  vec_t col_inv_count_;              // 1/(N_a·N_p·N_q) per column

  // ---- Histogram state ----
  mutable vec_t computed_;       // normalised, smoothed (length hist_len_)
  mutable vec_t smooth_scratch_; // reused smoothing buffer

  // Single-/multi-frame incremental histogram engine (shared with pair constraints).
  mutable IncrementalHistogram hist_;

  // One neighbour grid per frame (single-frame holds one); driven via update() hooks.
  mutable std::vector<NeighborGrid> frame_grids_;

  // ---- Optional drift-resync guard ----
  unsigned resync_every_{0};            // 0 = off; full rebuild every N accepts
  unsigned accepts_since_resync_{0};
};

static_assert(CConstraint<AngularDistributionConstraint>,
              "AngularDistributionConstraint must satisfy the CConstraint "
              "concept");

} // namespace RMC
