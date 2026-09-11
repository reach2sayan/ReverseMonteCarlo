#pragma once
#include <boost/describe/class.hpp>
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

// Symmetry-equivalent instances of one cluster type (orbit). Flat buffer:
// flat_sites holds body·instance_count site indices, instance i at [i·body,(i+1)·body).
struct ClusterOrbit {
  std::size_t body = 0;  // points per instance (0 ⇒ empty orbit)
  std::vector<std::size_t> flat_sites; // body * instance_count, contiguous
  double target = 0.0; // target correlation (0 = random equiatomic binary)
  double weight = 1.0; // weight in objective function
  // Multicomponent (ATAT) metadata, per cluster point. Empty ⇒ binary defaults
  // (site_type 0, func 0).
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

  // Basis coordinates of cluster point p. Empty metadata ⇒ binary defaults (0, 0).
  [[nodiscard]] constexpr int site_type_at(std::size_t p) const {
    return site_types.empty() ? 0 : site_types[p];
  }
  [[nodiscard]] constexpr int func_at(std::size_t p) const {
    return funcs.empty() ? 0 : funcs[p];
  }
};
BOOST_DESCRIBE_STRUCT(ClusterOrbit, (),
                      (body, flat_sites, target, weight, funcs, site_types))

// Site-basis table: one dense (n_func × n_occ) block per site type, read as
// table[site_type](func, occ). The ATAT pipeline keys blocks by sublattice id
// and addresses occupations by global label rank (zero outside the
// sublattice); the binary/linear setup is a single (1 × n) block of σ values.
using CorrFuncTable = std::vector<mat_t>;

// SQS-search constraint: weighted χ² deviation of multi-body cluster correlations
// from target. Reads only structure.elements (coordinates ignored). Site basis is
// either a SpeciesMap (element → scalar σ, binary/linear) or a CorrFuncTable +
// occupation-index map (full ATAT (m−1) trigonometric basis, m>2 components).
class ClusterCorrelationConstraint
    : public ConstraintBase<ClusterCorrelationConstraint> {
public:
  // Maps element symbol → occupation function value σ (typically ±1).
  using SpeciesMap = std::unordered_map<std::string, double>;

  // Binary / linear-encoding constructor: builds a degenerate single-function
  // table whose value per element is its scalar σ.
  ClusterCorrelationConstraint(const AtomicStructure &structure,
                               const SpeciesMap &species_map,
                               std::vector<ClusterOrbit> orbits);

  // Multicomponent constructor: explicit occupation-index map (element →
  // 0..m−1) and a CorrFuncTable (e.g. CorrFuncTable::trigonometric(m)).
  ClusterCorrelationConstraint(const AtomicStructure &structure,
                               std::unordered_map<std::string, int> occ_index,
                               CorrFuncTable table,
                               std::vector<ClusterOrbit> orbits);

  // Full from-scratch error (oracle; used by current_correlations() and resync).
  [[nodiscard]] double compute_error(const coords_t &,
                                     std::span<const std::size_t>);

  // --- Incremental interface (overrides ConstraintBase defaults) -----------
  // Maintains a running per-orbit raw sum + total error, updating only instances
  // touching a changed site. Float drift bounded by periodic full resync on accept().
  void compute_before_move(const coords_t &,
                           std::span<const std::size_t>);
  void compute_after_move(const coords_t &,
                          std::span<const std::size_t>);
  void accept() noexcept;
  void reject() noexcept;

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "ClusterCorrelation";
  }

  // More expensive than cheap geometric constraints.
  [[nodiscard]] constexpr double
  computation_cost() const noexcept {
    return static_cast<double>(total_instances_) * 10.0;
  }

  [[nodiscard]] std::vector<double> current_correlations() const;
  [[nodiscard]] constexpr const std::vector<ClusterOrbit> &
  orbits() const noexcept {
    return orbits_;
  }

private:
  // Runtime state for one orbit, by orbit index.
  struct OrbitState {
    std::size_t base = 0;   // global instance id of this orbit's first instance
    std::size_t count = 0;  // instance count (cached orbit.instance_count())
    double sum = 0.0;       // running raw Σ of per-instance products
    std::uint64_t seen = 0; // per-step epoch dedup stamp
  };

  // Static adjacency over (orbit, gid, site) id-spaces, built once by build_index():
  // gid→orbit map and site→instances reverse index.
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

  void build_occ_of_code();
  void refresh_site_occ() const;
  [[nodiscard]] double orbit_correlation(const ClusterOrbit &orbit) const;
  [[nodiscard]] int current_occ(std::size_t k) const;
  // Product of the basis values over one instance's sites, reading each site's
  // occupation from `occ` (an unknown species, -1, zeroes the product).
  [[nodiscard]] double product(const ClusterOrbit &orb,
                               std::span<const std::size_t> sites,
                               std::span<const int> occ) const;
  [[nodiscard]] double instance_product(std::uint32_t gid) const;
  // Sites of one instance (global id) as a view into its orbit's flat buffer.
  [[nodiscard]] std::span<const std::size_t> sites_of(std::uint32_t gid) const;
  void build_index();
  void resync_full();
  void ensure_built();
  [[nodiscard]] constexpr bool occ_mismatch() const;
  // Incremental move application: diff the changed sites, then (if any)
  // collect the touched instances/orbits, log undo info, commit the new
  // occupations and patch the affected orbit sums and the total error.
  void apply_move_update();
  void collect_affected();     // touched instances/orbits (epoch dedup)
  void save_undo();            // undo log + pre-move per-instance prods
  void apply_product_deltas(); // patch affected orbit sums
  void patch_orbit_errors();   // patch total_err_ for those orbits
  void rollback_move() noexcept;

  const AtomicStructure &structure_;
  std::unordered_map<std::string, int> occ_index_; // element → occupation index
  // Flat code → occupation-index table (integer fast path in refresh_site_occ());
  // empty ⇒ fall back to occ_index_ string lookup.
  std::vector<int> occ_of_code_;
  CorrFuncTable table_;
  std::vector<ClusterOrbit> orbits_;
  // Per-site occupation index, refreshed per evaluation from structure_.elements;
  // -1 = unknown species. mutable: refreshed by const current_correlations() too.
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
  // Per-step dedup of touched instances via epoch stamp (per-orbit in OrbitState::seen).
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

} // namespace RMC
