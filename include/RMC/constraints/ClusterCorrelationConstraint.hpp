#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/core/Structure.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace RMC {

// All symmetry-equivalent instances of one cluster type (orbit).
//
// Every instance shares the same arity (`body` = cluster point count), so the
// instances are stored as a single contiguous flat buffer rather than a
// vector-of-vectors: `flat_sites` holds body·instance_count supercell site
// indices, instance i occupying [i·body, (i+1)·body). Read them as a range of
// contiguous subranges via instances() (the index order within each subrange
// matches the orbit's per-point func/site_type).
struct ClusterOrbit {
  std::size_t body = 0;  // points per instance (0 ⇒ empty orbit)
  std::vector<std::size_t> flat_sites; // body * instance_count, contiguous
  double target = 0.0; // target correlation (0 = random equiatomic binary)
  double weight = 1.0; // weight in objective function
  // Multicomponent (ATAT) metadata, per cluster point. Empty ⇒ binary defaults
  // (site_type 0, func 0) so legacy binary orbits keep working unchanged.
  std::vector<int> funcs;
  std::vector<int> site_types;

  [[nodiscard]] constexpr std::size_t instance_count() const noexcept {
    return body == 0 ? 0 : flat_sites.size() / body;
  }

  // Range-like access: a view of contiguous subranges, one per instance.
  [[nodiscard]] constexpr auto instances() const {
    return flat_sites | std::views::chunk(body);
  }

  // Append one instance; the first call fixes `body`.
  void add_instance(std::span<const std::size_t> sites) {
    if (body == 0) {
      body = sites.size();
    }
    flat_sites.insert(flat_sites.end(), sites.begin(), sites.end());
  }

  // Replace all instances from literal site tuples (e.g. {{0,1},{1,2}}).
  void set_instances(
      std::initializer_list<std::initializer_list<std::size_t>> insts) {
    body = 0;
    flat_sites.clear();
    for (const auto &sites : insts) {
      add_instance(std::span<const std::size_t>(sites.begin(), sites.size()));
    }
  }

  // Basis coordinates of cluster point p. Empty metadata ⇒ binary defaults
  // (site_type 0, func 0), so legacy binary orbits index the degenerate table.
  [[nodiscard]] constexpr int site_type_at(std::size_t p) const {
    return site_types.empty() ? 0 : site_types[p];
  }
  [[nodiscard]] constexpr int func_at(std::size_t p) const {
    return funcs.empty() ? 0 : funcs[p];
  }
};

// ATAT trigonometric (Chebyshev-like) site-basis table, addressed by the
// coordinate (site_type, func, occupation), where site_type = (#components − 2).
// For binary it reduces to {occ 0 → −1, occ 1 → +1}. Port of
// TrigoCorrFuncTable::init (atat/src/calccorr.c++:163-180).
//
// Storage is one flat buffer holding a rectangular (n_func × n_occ) block per
// site_type, reached through the C++23 multidimensional subscript — so callers
// write table[site_type, func, occ] instead of casting three indices into a
// nested vector-of-vector-of-vector. Build by appending one block per site_type
// with add_site_type(), then filling cells through the mutable subscript.
class CorrFuncTable {
public:
  // Append a zero-initialised (n_func × n_occ) block; returns its site_type
  // index (== number of prior site_types). Call once per site_type, in order.
  std::size_t add_site_type(int n_func, int n_occ) {
    blocks_.push_back({data_.size(), n_occ});
    data_.resize(data_.size() + static_cast<std::size_t>(n_func) *
                                    static_cast<std::size_t>(n_occ));
    return blocks_.size() - 1;
  }

  // Portable accessors (work on every C++23 compiler): value() reads, at()
  // returns a mutable cell for building. These are the canonical API used
  // internally — the operator[] below is sugar that forwards to them.
  [[nodiscard]] constexpr double value(int site_type, int func,
                                       int occ) const {
    return data_[flat_index(site_type, func, occ)];
  }
  [[nodiscard]] constexpr double &at(int site_type, int func, int occ) {
    return data_[flat_index(site_type, func, occ)];
  }

  // C++23 multidimensional subscript: table[site_type, func, occ]. Guarded by
  // its feature-test macro so the declaration itself is skipped on toolchains
  // without P2128 (e.g. MSVC < 17.10); value()/at() remain the safe fallback.
#ifdef __cpp_multidimensional_subscript
  [[nodiscard]] constexpr double operator[](int site_type, int func,
                                            int occ) const {
    return value(site_type, func, occ);
  }
  [[nodiscard]] constexpr double &operator[](int site_type, int func, int occ) {
    return at(site_type, func, occ);
  }
#endif

  [[nodiscard]] constexpr std::size_t site_type_count() const noexcept {
    return blocks_.size();
  }

  [[nodiscard]] static CorrFuncTable trigonometric(int max_components);

private:
  // One site_type's rectangular block: where it starts in data_ and its row
  // stride (n_occ). n_func is implied by the gap to the next block.
  struct Block {
    std::size_t offset;
    int n_occ;
  };
  [[nodiscard]] constexpr std::size_t flat_index(int site_type, int func,
                                                 int occ) const {
    const Block &b = blocks_[static_cast<std::size_t>(site_type)];
    return b.offset +
           static_cast<std::size_t>(func) * static_cast<std::size_t>(b.n_occ) +
           static_cast<std::size_t>(occ);
  }

  std::vector<Block> blocks_; // per site_type
  std::vector<double> data_;  // flat (n_func × n_occ) blocks, concatenated
};

// Constraint for Special Quasi-random Structure (SQS) search.
//
// Tracks the weighted χ² deviation of multi-body cluster correlations from
// target values. Works with SpeciesSwapGenerator: coordinates are ignored; only
// structure.elements (species labels) are read.
//
// Two ways to define the site basis:
//   • SpeciesMap — element → scalar σ (binary / simple linear encoding):
//       {{"Cu", +1.0}, {"Au", -1.0}}
//   • CorrFuncTable + occupation-index map — the full ATAT (m−1) trigonometric
//     basis for correct multicomponent (m>2) correlations; supplied by the
//     ClusterEnumerator from corrdump's clusters.out.
//
// Cluster instances are pre-computed (ATAT corrdump + symmetry enumeration) and
// passed in as ClusterOrbit objects.
class ClusterCorrelationConstraint
    : public ConstraintBase<ClusterCorrelationConstraint> {
public:
  // Maps element symbol → occupation function value σ (typically ±1).
  using SpeciesMap = std::unordered_map<std::string, double>;

  // Binary / linear-encoding constructor (back-compatible). Builds a degenerate
  // single-function table whose value for each element is its scalar σ.
  ClusterCorrelationConstraint(const AtomicStructure &structure,
                               const SpeciesMap &species_map,
                               std::vector<ClusterOrbit> orbits);

  // Multicomponent constructor: explicit occupation-index map (element →
  // 0..m−1) and a CorrFuncTable (e.g. CorrFuncTable::trigonometric(m)).
  ClusterCorrelationConstraint(const AtomicStructure &structure,
                               std::unordered_map<std::string, int> occ_index,
                               CorrFuncTable table,
                               std::vector<ClusterOrbit> orbits);

  // Full from-scratch error (oracle / used by current_correlations() and the
  // periodic resync). The engine no longer routes through this — it uses the
  // incremental compute_before_move/compute_after_move overrides below.
  [[nodiscard]] double compute_error(const coords_t &,
                                     std::span<const std::size_t>);

  // --- Incremental interface (overrides ConstraintBase defaults) -----------
  // A SpeciesSwap move changes the occupation of only ~2 sites, so instead of
  // refolding every orbit instance (O(total_instances)) we maintain a running
  // per-orbit raw sum + total error and update only the instances that touch a
  // changed site (O(N scan + affected_instances)). Float drift is bounded by a
  // periodic full resync on accept().
  void compute_before_move(Constraint::Token, const coords_t &,
                           std::span<const std::size_t>);
  void compute_after_move(Constraint::Token, const coords_t &,
                          std::span<const std::size_t>);
  void accept(Constraint::Token) noexcept;
  void reject(Constraint::Token) noexcept;

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "ClusterCorrelation";
  }

  // More expensive than cheap geometric constraints.
  [[nodiscard]] constexpr double
  computation_cost(Constraint::Token) const noexcept {
    return static_cast<double>(total_instances_) * 10.0;
  }

  [[nodiscard]] std::vector<double> current_correlations() const;
  [[nodiscard]] constexpr const std::vector<ClusterOrbit> &
  orbits() const noexcept {
    return orbits_;
  }

private:
  // Runtime state for one orbit, parallel to orbits_ by orbit index. Collapses
  // what used to be four index-aligned vectors (base/count/sum/seen) into one
  // struct so the per-orbit fields can never drift out of alignment.
  struct OrbitState {
    std::size_t base = 0;   // global instance id of this orbit's first instance
    std::size_t count = 0;  // instance count (cached orbit.instance_count())
    double sum = 0.0;       // running raw Σ of per-instance products
    std::uint64_t seen = 0; // per-step epoch dedup stamp
  };

  // Static adjacency over the (orbit, global-instance-id, site) id-spaces, built
  // once by build_index(). Owns the gid→orbit map and the site→instances
  // reverse index and exposes them through named accessors, so the accounting
  // code never hand-rolls id arithmetic.
  struct ClusterIndex {
    std::vector<std::uint32_t> inst_orbit;  // global instance id → orbit index
    std::vector<std::vector<std::uint32_t>> site_to_instances; // site → gids

    [[nodiscard]] std::uint32_t orbit_of(std::uint32_t gid) const {
      return inst_orbit[gid];
    }
    [[nodiscard]] std::span<const std::uint32_t>
    instances_of_site(std::size_t site) const {
      return site_to_instances[site];
    }
    [[nodiscard]] std::size_t total_instances() const {
      return inst_orbit.size();
    }
    [[nodiscard]] std::size_t site_count() const {
      return site_to_instances.size();
    }
  };

  // Per-move pipeline context threaded through the apply_move_update() and_then
  // chain (mirrors EngineBase::TrialCtx). The heavy scratch buffers stay as
  // members; this only carries the changed-site view that seeds later steps.
  struct MoveCtx {
    std::span<const std::size_t> changed; // view into changed_sites_
  };

  void build_occ_of_code();
  void refresh_site_occ() const;
  [[nodiscard]] double orbit_correlation(const ClusterOrbit &orbit) const;
  [[nodiscard]] int current_occ(std::size_t k) const;
  [[nodiscard]] double instance_product(std::uint32_t gid) const;
  // Sites of one instance (global id) as a view into its orbit's flat buffer.
  [[nodiscard]] std::span<const std::size_t> sites_of(std::uint32_t gid) const;
  void build_index();
  void resync_full();
  void ensure_built();
  [[nodiscard]] constexpr bool occ_mismatch() const;
  // Incremental move application, expressed as a monadic and_then pipeline of
  // the private steps below. diff_changed_sites() seeds the chain and
  // short-circuits it (empty optional) when the move changed no occupation.
  void apply_move_update();
  [[nodiscard]] std::optional<MoveCtx> diff_changed_sites();
  void collect_affected(MoveCtx &c);   // touched instances/orbits (epoch dedup)
  void save_undo(MoveCtx &c);          // undo log + pre-move per-instance prods
  void commit_occupations(MoveCtx &c); // write new occ_ for changed sites
  void apply_product_deltas(const MoveCtx &); // patch affected orbit sums
  void patch_orbit_errors(const MoveCtx &);   // patch total_err_ for those orbits
  void rollback_move() noexcept;

  const AtomicStructure &structure_;
  std::unordered_map<std::string, int> occ_index_; // element → occupation index
  // Flat occupation-index → occupation-index table for the integer fast path in
  // refresh_site_occ(); empty ⇒ fall back to the occ_index_ string lookup.
  std::vector<int> occ_of_code_;
  CorrFuncTable table_;
  std::vector<ClusterOrbit> orbits_;
  // Per-site occupation index, refreshed once per evaluation from
  // structure_.elements; -1 = unknown species. Source of the hot-loop reads.
  // mutable: refreshed by const current_correlations() as well as
  // compute_error.
  mutable std::vector<int> site_occ_;
  std::size_t total_instances_ = [this] {
    return std::transform_reduce(
        orbits_.begin(), orbits_.end(), std::size_t{0}, std::plus<>{},
        [](const auto &o) { return o.instance_count(); });
  }();

  // --- Incremental state (built lazily by ensure_built) --------------------
  bool built_ = false;
  std::vector<int> occ_;           // committed per-site occupation index
  double total_err_ = 0.0;         // running Σ weight·(corr−target)²
  ClusterIndex index_;             // gid↔orbit + site→gids adjacency (built once)
  std::vector<OrbitState> ostate_; // per-orbit runtime state, by orbit index
  // Per-step dedup of touched instances via an epoch stamp (per-orbit dedup
  // lives in OrbitState::seen).
  std::uint64_t epoch_ = 0;
  std::vector<std::uint64_t> seen_inst_; // per-gid stamp
  // Per-step scratch (members to avoid reallocation each step).
  std::vector<std::size_t> changed_sites_;
  std::vector<std::uint32_t> affected_insts_;
  std::vector<std::size_t> affected_orbits_;
  std::vector<double> old_prod_;
  // Undo log for reject(): changed sites' old occ, affected orbits' old sums.
  std::vector<std::pair<std::size_t, int>> undo_sites_;
  std::vector<std::pair<std::size_t, double>> undo_orbit_sums_;
  // Periodic full resync to bound floating-point drift from incremental sums.
  std::size_t accepted_since_resync_ = 0;
  static constexpr std::size_t kResyncInterval = 4096;
};

static_assert(
    CConstraint<ClusterCorrelationConstraint>,
    "ClusterCorrelationConstraint must satisfy the CConstraint concept");

} // namespace RMC
