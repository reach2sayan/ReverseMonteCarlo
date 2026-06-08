#pragma once
#include <RMC/constraints/Constraint.hpp>
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
// the upper-triangular range [0, adf_n_leg_pairs). Mirrors the packing MAST
// uses for its angle triplet types (Angles::get_type), so column order is
// deterministic.
[[nodiscard]] constexpr FORCE_INLINE int
adf_leg_pair_index(int p, int q, int n_types) noexcept {
  return p * (2 * n_types - p + 1) / 2 + (q - p);
}

// Total number of triplet columns: central element × unordered leg pair.
[[nodiscard]] constexpr FORCE_INLINE int adf_n_cols(int n_types) noexcept {
  return n_types * adf_n_leg_pairs(n_types);
}

// Accumulate a raw bond-angle histogram into `hist` (length n_bins ·
// adf_n_cols). For every central atom i, every unordered pair (j, k) of its
// neighbours within `max_dis` contributes 1 to the bin of the angle j–i–k
// (binned over [0, π] into n_bins) and the column (central = elem_id[i], legs =
// elem_id[j], elem_id[k]). `hist` is zeroed first. `elem_id` holds the small
// contiguous species id per atom (the same 0,1,2,… ids initialise() assigns),
// `n_types` their count.
//
// This is a full O(N·⟨n⟩²) recompute, matching MAST's reference RMC (which
// rebuilds the whole ADF every step). It targets the small DFT-ready cells the
// Special Glass Structure method produces; an incremental neighbour-delta path
// (mirroring accumulate_moved_pairs) is the planned follow-up optimisation.
void accumulate_angle_histogram(vec_t &hist, const coords_t &coords,
                                const BoundaryConditions *bc,
                                const std::vector<uint8_t> &elem_id,
                                int n_types, double max_dis, int n_bins,
                                const AtomsCollector *collector = nullptr);

// Incremental neighbour-delta companion to accumulate_angle_histogram. Adds
// into `hist` (caller zeroes) every angle j–i–k PRESENT in `coords` whose
// vertex set {apex i, leg j, leg k} intersects `moved`, each exactly once, in
// the SAME bin/column layout. `grid` must already be synced to `coords` (it
// serves the neighbour queries). Used for the before/after-move delta:
// new_hist = saved − accumulate_moved_angles(old) + accumulate_moved_angles(new).
//
// Iterating by apex over the candidate centres { moved ∪ neighbours(moved) }
// counts each affected angle exactly once (the apex is its unique key — no
// ordering trick is needed, unlike accumulate_moved_pairs). The seed reads a
// moved atom's coordinate row even when it is absent, so a removed atom's
// former neighbours are still revisited (its angles get subtracted, not
// re-added). Per-step cost is independent of N.
void accumulate_moved_angles(vec_t &hist, const coords_t &coords,
                             const BoundaryConditions *bc,
                             const std::vector<uint8_t> &elem_id, int n_types,
                             double max_dis, int n_bins,
                             std::span<const std::size_t> moved,
                             const NeighborGrid &grid,
                             const AtomsCollector *collector = nullptr);

// Soft constraint fitting the bond-angle distribution function (ADF) — the
// angular companion to PairDistributionConstraint. Distinct from
// AngleConstraint (a rigid per-triplet bound check): this fits the full angular
// *distribution* histogram, per element triplet, against a target, contributing
// to the engine's total chi².
//
// The histogram is one flat vec_t, row-major angle_bin × triplet_column (stride
// = n_cols_), so the multi-frame averaging arithmetic carries over verbatim
// from PairConstraintBase. Element ids are assigned by *sorted* symbol (as in
// analysis::compute_adf and compute_gr) so the column order is independent of
// the structure's atom ordering and a target written by `--adf-compute` is
// directly consumable here.
class AngularDistributionConstraint
    : public SingularConstraintBase<AngularDistributionConstraint> {
public:
  using SingularConstraintBase<
      AngularDistributionConstraint>::set_boundary_conditions;

  // Experimental target. Column 0 = angle bin centres (rad, over [0, π]); each
  // remaining column = one triplet's target value, in the canonical column
  // order (central id outer, leg pair inner — see adf_leg_pair_index). The
  // number of value columns must equal adf_n_cols(n_types); validated in
  // initialise().
  void set_experimental_data(const mat_t &data);

  constexpr void set_elements(std::span<const std::string> e) noexcept {
    elements_ = e;
  }
  constexpr void set_cutoff(double max_dis) noexcept { max_dis_ = max_dis; }
  constexpr void set_smoothing(int range) noexcept { smooth_range_ = range; }
  // When true (default) the chi² is RMC's scale-invariant residual, matching
  // the pair constraints. Set false for MAST-parity (unscaled L2 + full MAST
  // volume normalization) when validating against the reference code.
  constexpr void set_scale_invariant(bool v) noexcept { scale_invariant_ = v; }

  // Force a full histogram + grid rebuild every `n` accepted moves to bound the
  // floating-point drift of the incremental `saved − D_old + D_new` update. 0
  // (default) disables it, matching the never-resyncing pair constraints; the
  // per-step deltas are sums of integer counts so drift is negligible in
  // practice.
  constexpr void set_resync_interval(unsigned n) noexcept { resync_every_ = n; }

  // build species ids, per-column normalization, and flatten/validate target.
  // Call after set_experimental_data and set_elements (automatic).
  void initialise();

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "AngularDistributionConstraint";
  }
  // Between PDF (1e6) and S(Q) (2e6): a full ADF recompute is O(N·⟨n⟩²), so it
  // runs after the PDF in the cost-ordered collection.
  [[nodiscard]] static constexpr double
  computation_cost(Constraint::Token) noexcept {
    return 1.5e6;
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

  // Public (non-token) multi-frame controls, mirroring PairConstraintBase so
  // tests and tools can drive them directly. set_active_frame_idx resets the
  // pending before-move delta: the saved snapshot/grid belong to the previously
  // active frame, so a before-move on a switched-to frame must not be mistaken
  // for an after-move.
  void set_n_frames(std::size_t n);
  void set_active_frame_idx(std::size_t k) noexcept {
    active_frame_ = k;
    incremental_ready_ = false;
  }
  // Restore the active frame's histogram + grid after a rejected move; commit
  // drops the rollback snapshot (and drives the optional drift resync).
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

  // The neighbour grid backing the active frame (grid_ in single-frame mode).
  [[nodiscard]] NeighborGrid &active_grid() const noexcept {
    return n_frames_ > 1 ? frame_grids_[active_frame_] : grid_;
  }

  // ---- Configuration ----
  vec_t exp_theta_;     // angle bin centres (col 0 of the data file)
  mat_t exp_values_;    // staged value columns, flattened in initialise()
  vec_t exp_data_;      // flattened target, length hist_len_
  double max_dis_{3.4}; // bond cutoff (Å)
  int n_bins_{0};       // angle bins over [0, π]
  int smooth_range_{2}; // boxcar half-width (0 disables); 2 passes, MAST-style
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
  mutable vec_t single_hist_;    // raw counts, single-frame path
  mutable bool single_hist_current_{false}; // false ⇒ next call full-rebuilds
  mutable vec_t smooth_scratch_; // reused smoothing buffer

  // ---- Multi-frame state (mutable: modified inside const compute_error) ----
  std::size_t n_frames_{1};
  std::size_t active_frame_{0};
  mutable std::vector<vec_t> frame_hists_;       // raw counts, per frame
  mutable std::vector<bool> frame_hist_current_; // per-frame one-time-build gate
  mutable vec_t sum_hist_;                        // running Σ over frames
  mutable vec_t saved_frame_hist_;                // pre-move snapshot for rollback

  // ---- Incremental neighbour-delta state ----
  // The before-move call records the moved-vertex angle contribution at the old
  // coords into saved_moved_delta_; the after-move call recomputes it at the new
  // coords into scratch_delta_ and patches saved_frame_hist_ ± deltas, exactly
  // as PairConstraintBase does with accumulate_moved_pairs.
  mutable vec_t saved_moved_delta_;
  mutable vec_t scratch_delta_;
  mutable bool incremental_ready_{false}; // set by before-move, cleared after
  mutable NeighborGrid grid_;             // single-frame neighbour grid
  mutable std::vector<NeighborGrid> frame_grids_; // per-frame neighbour grids

  // ---- Optional drift-resync guard ----
  unsigned resync_every_{0};            // 0 = off; full rebuild every N accepts
  unsigned accepts_since_resync_{0};
};

static_assert(CConstraint<AngularDistributionConstraint>,
              "AngularDistributionConstraint must satisfy the CConstraint "
              "concept");

} // namespace RMC
