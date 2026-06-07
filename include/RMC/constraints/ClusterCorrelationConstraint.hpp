#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/core/Structure.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
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
  std::size_t body = 0;                // points per instance (0 ⇒ empty orbit)
  std::vector<std::size_t> flat_sites; // body * instance_count, contiguous
  double target = 0.0; // target correlation (0 = random equiatomic binary)
  double weight = 1.0; // weight in objective function
  // Multicomponent (ATAT) metadata, per cluster point. Empty ⇒ binary defaults
  // (site_type 0, func 0) so legacy binary orbits keep working unchanged.
  std::vector<int> funcs;
  std::vector<int> site_types;

  [[nodiscard]] std::size_t instance_count() const noexcept {
    return body == 0 ? 0 : flat_sites.size() / body;
  }

  // Range-like access: a view of contiguous subranges, one per instance.
  [[nodiscard]] auto instances() const {
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
};

// ATAT trigonometric (Chebyshev-like) site-basis table: value indexed by
// [site_type][func][occupation], where site_type = (#components − 2). For binary
// it reduces to {occ 0 → −1, occ 1 → +1}. Port of TrigoCorrFuncTable::init
// (atat/src/calccorr.c++:163-180).
struct CorrFuncTable {
  std::vector<std::vector<std::vector<double>>> t; // [site_type][func][occ]

  [[nodiscard]] double value(int site_type, int func, int occ) const {
    return t[static_cast<std::size_t>(site_type)][static_cast<std::size_t>(func)]
            [static_cast<std::size_t>(occ)];
  }

  [[nodiscard]] static CorrFuncTable trigonometric(int max_components) {
    CorrFuncTable tab;
    if (max_components < 2) {
      max_components = 2;
    }
    tab.t.resize(static_cast<std::size_t>(max_components - 1));
    for (int m = 2; m <= max_components; ++m) {
      auto &site = tab.t[static_cast<std::size_t>(m - 2)];
      site.resize(static_cast<std::size_t>(m - 1));
      for (int f = 0; f < m - 1; ++f) {
        site[static_cast<std::size_t>(f)].assign(static_cast<std::size_t>(m),
                                                 0.0);
      }
      for (int s = 0; s < m; ++s) {
        for (int k = 1; k <= m / 2; ++k) {
          site[static_cast<std::size_t>(2 * k - 2)][static_cast<std::size_t>(s)] =
              -std::cos(2.0 * M_PI * s * k / m);
        }
        for (int k = 1; k <= (m + 1) / 2 - 1; ++k) {
          site[static_cast<std::size_t>(2 * k - 1)][static_cast<std::size_t>(s)] =
              -std::sin(2.0 * M_PI * s * k / m);
        }
      }
    }
    return tab;
  }
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
                               std::vector<ClusterOrbit> orbits)
      : structure_(structure), orbits_(std::move(orbits)) {
    table_.t.resize(1);
    table_.t[0].resize(1);
    table_.t[0][0].resize(species_map.size(), 0.0);
    for (const auto [i, kv] : std::views::enumerate(species_map)) {
      const auto &[elem, sigma] = kv;
      occ_index_[elem] = static_cast<int>(i);
      table_.t[0][0][static_cast<std::size_t>(i)] = sigma;
    }
  }

  // Multicomponent constructor: explicit occupation-index map (element →
  // 0..m−1) and a CorrFuncTable (e.g. CorrFuncTable::trigonometric(m)).
  ClusterCorrelationConstraint(const AtomicStructure &structure,
                               std::unordered_map<std::string, int> occ_index,
                               CorrFuncTable table,
                               std::vector<ClusterOrbit> orbits)
      : structure_(structure), occ_index_(std::move(occ_index)),
        table_(std::move(table)), orbits_(std::move(orbits)) {
    build_occ_of_code();
  }

  // Full from-scratch error (oracle / used by current_correlations() and the
  // periodic resync). The engine no longer routes through this — it uses the
  // incremental compute_before_move/compute_after_move overrides below.
  [[nodiscard]] double compute_error(const coords_t &,
                                     std::span<const std::size_t>) {
    refresh_site_occ();
    return std::ranges::fold_left(
        orbits_, 0.0, [this](double total, const ClusterOrbit &orbit) {
          const double dev = orbit_correlation(orbit) - orbit.target;
          return total + orbit.weight * dev * dev;
        });
  }

  // --- Incremental interface (overrides ConstraintBase defaults) -----------
  // A SpeciesSwap move changes the occupation of only ~2 sites, so instead of
  // refolding every orbit instance (O(total_instances)) we maintain a running
  // per-orbit raw sum + total error and update only the instances that touch a
  // changed site (O(N scan + affected_instances)). Float drift is bounded by a
  // periodic full resync on accept().
  void compute_before_move(Constraint::Token, const coords_t &,
                           std::span<const std::size_t>) {
    ensure_built();
    // Absorb any external structure replacement (e.g. cooperative-ensemble
    // broadcast) into the committed baseline before scoring. A normal step
    // changes nothing here (the move hasn't happened yet) — just an O(N) scan.
    if (occ_mismatch()) {
      resync_full();
    }
    err_before_ = total_err_;
    undo_sites_.clear();
    undo_orbit_sums_.clear();
  }
  void compute_after_move(Constraint::Token, const coords_t &,
                          std::span<const std::size_t>) {
    apply_move_update();
    err_after_ = total_err_;
  }
  void accept(Constraint::Token) noexcept {
    err_before_ = err_after_;
    undo_sites_.clear();
    undo_orbit_sums_.clear();
    if (++accepted_since_resync_ >= kResyncInterval) {
      resync_full(); // re-sync state vectors are pre-sized: no allocation
      accepted_since_resync_ = 0;
      err_before_ = err_after_ = total_err_;
    }
  }
  void reject(Constraint::Token) noexcept {
    rollback_move();
    total_err_ = err_before_;
    err_after_ = err_before_;
  }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "ClusterCorrelation";
  }

  // More expensive than cheap geometric constraints.
  [[nodiscard]] double computation_cost(Constraint::Token) const noexcept {
    return static_cast<double>(total_instances_) * 10.0;
  }

  [[nodiscard]] std::vector<double> current_correlations() const {
    refresh_site_occ();
    auto corr = orbits_ | std::views::transform([this](const auto &orbit) {
                  return orbit_correlation(orbit);
                });
    return {corr.begin(), corr.end()};
  }

  [[nodiscard]] const std::vector<ClusterOrbit> &orbits() const noexcept {
    return orbits_;
  }

private:
  // Build the flat occupation-index → occupation-index table used by the hot
  // path. In the multicomponent (ATAT) setup structure_.atomic_numbers already
  // stores the occupation index per site (ClusterEnumerator sets
  // atomic_numbers[a] = occ_index[elements[a]]), so the table is the identity
  // over the valid occupation indices with -1 in any gap. Empty table ⇒ the
  // string fallback in refresh_site_occ() is used (binary / test path, where
  // atomic_numbers carry true Z rather than occupation indices).
  void build_occ_of_code() {
    int max_occ = -1;
    for (const auto &[elem, occ] : occ_index_) {
      max_occ = std::max(max_occ, occ);
    }
    if (max_occ < 0) {
      return;
    }
    occ_of_code_.assign(static_cast<std::size_t>(max_occ) + 1, -1);
    for (const auto &[elem, occ] : occ_index_) {
      occ_of_code_[static_cast<std::size_t>(occ)] = occ;
    }
  }

  // Resolve every site's element → occupation index once per evaluation, so the
  // hot orbit loop indexes a flat int array instead of hashing/comparing the
  // std::string element symbol per cluster point. -1 marks an unknown species.
  //
  // Fast path (occ_of_code_ populated): gather the occupation index straight
  // from the integer structure_.atomic_numbers, avoiding the per-site string
  // hash + memcmp that dominated the SQS hot loop. Falls back to the string map
  // when atomic_numbers do not encode occupation indices.
  void refresh_site_occ() const {
    const std::size_t n = structure_.elements.size();
    site_occ_.resize(n);
    if (!occ_of_code_.empty()) {
      const int cap = static_cast<int>(occ_of_code_.size());
      for (std::size_t s = 0; s < n; ++s) {
        const int code = structure_.atomic_numbers[static_cast<Eigen::Index>(s)];
        site_occ_[s] = (code >= 0 && code < cap) ? occ_of_code_[static_cast<std::size_t>(code)] : -1;
      }
      return;
    }
    for (std::size_t s = 0; s < n; ++s) {
      const auto it = occ_index_.find(structure_.elements[s]);
      site_occ_[s] = (it == occ_index_.end()) ? -1 : it->second;
    }
  }

  [[nodiscard]] double orbit_correlation(const ClusterOrbit &orbit) const {
    const std::size_t count = orbit.instance_count();
    if (count == 0) {
      return 0.0;
    }
    double sum = 0.0;
    for (const auto inst : orbit.instances()) {
      double prod = 1.0;
      for (const auto [p, site] : std::views::enumerate(inst)) {
        const int occ = site_occ_[site];
        if (occ < 0) {
          prod = 0.0; // unknown species → zero contribution (legacy behaviour)
          break;
        }
        const int st = orbit.site_types.empty() ? 0 : orbit.site_types[p];
        const int fn = orbit.funcs.empty() ? 0 : orbit.funcs[p];
        prod *= table_.value(st, fn, occ);
      }
      sum += prod;
    }
    return sum / static_cast<double>(count);
  }

  // --- Incremental machinery -----------------------------------------------

  // Current occupation index of site k from the live structure (mirrors the two
  // paths in refresh_site_occ but for a single site).
  [[nodiscard]] int current_occ(std::size_t k) const {
    if (!occ_of_code_.empty()) {
      const int code = structure_.atomic_numbers[static_cast<Eigen::Index>(k)];
      const int cap = static_cast<int>(occ_of_code_.size());
      return (code >= 0 && code < cap)
                 ? occ_of_code_[static_cast<std::size_t>(code)]
                 : -1;
    }
    const auto it = occ_index_.find(structure_.elements[k]);
    return it == occ_index_.end() ? -1 : it->second;
  }

  // Product of the basis values over one instance (global id), using occ_.
  [[nodiscard]] double instance_product(std::uint32_t gid) const {
    const std::size_t o = inst_orbit_[gid];
    const ClusterOrbit &orb = orbits_[o];
    const std::size_t off =
        (static_cast<std::size_t>(gid) - orbit_base_[o]) * orb.body;
    double prod = 1.0;
    for (std::size_t p = 0; p < orb.body; ++p) {
      const int occ = occ_[orb.flat_sites[off + p]];
      if (occ < 0) {
        return 0.0;
      }
      const int st = orb.site_types.empty() ? 0 : orb.site_types[p];
      const int fn = orb.funcs.empty() ? 0 : orb.funcs[p];
      prod *= table_.value(st, fn, occ);
    }
    return prod;
  }

  // Build the site→instance reverse index and per-orbit globals (once).
  void build_index() {
    const std::size_t n =
        std::max(structure_.elements.size(),
                 static_cast<std::size_t>(structure_.atomic_numbers.size()));
    orbit_base_.resize(orbits_.size());
    orbit_count_.resize(orbits_.size());
    std::size_t total = 0;
    for (std::size_t o = 0; o < orbits_.size(); ++o) {
      orbit_base_[o] = total;
      orbit_count_[o] = orbits_[o].instance_count();
      total += orbit_count_[o];
    }
    inst_orbit_.assign(total, 0);
    seen_inst_.assign(total, 0);
    seen_orbit_.assign(orbits_.size(), 0);
    orbit_sum_.assign(orbits_.size(), 0.0);
    site_to_instances_.assign(n, {});
    for (std::size_t o = 0; o < orbits_.size(); ++o) {
      const ClusterOrbit &orb = orbits_[o];
      for (std::size_t li = 0; li < orbit_count_[o]; ++li) {
        const auto gid = static_cast<std::uint32_t>(orbit_base_[o] + li);
        inst_orbit_[gid] = static_cast<std::uint32_t>(o);
        const std::size_t off = li * orb.body;
        for (std::size_t p = 0; p < orb.body; ++p) {
          const std::size_t site = orb.flat_sites[off + p];
          if (site < site_to_instances_.size()) {
            site_to_instances_[site].push_back(gid);
          }
        }
      }
    }
  }

  // Recompute occ_, orbit_sum_ and total_err_ from scratch (init + resync).
  void resync_full() {
    const std::size_t n = site_to_instances_.size();
    occ_.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
      occ_[k] = current_occ(k);
    }
    total_err_ = 0.0;
    for (std::size_t o = 0; o < orbits_.size(); ++o) {
      double sum = 0.0;
      for (std::size_t li = 0; li < orbit_count_[o]; ++li) {
        sum += instance_product(static_cast<std::uint32_t>(orbit_base_[o] + li));
      }
      orbit_sum_[o] = sum;
      if (orbit_count_[o] == 0) {
        continue;
      }
      const double dev =
          sum / static_cast<double>(orbit_count_[o]) - orbits_[o].target;
      total_err_ += orbits_[o].weight * dev * dev;
    }
  }

  void ensure_built() {
    if (built_) {
      return;
    }
    build_index();
    resync_full();
    built_ = true;
  }

  // True if the live structure no longer matches the committed occ_ baseline.
  [[nodiscard]] bool occ_mismatch() const {
    for (std::size_t k = 0; k < occ_.size(); ++k) {
      if (current_occ(k) != occ_[k]) {
        return true;
      }
    }
    return false;
  }

  // Apply the just-proposed move: diff changed sites, update only the affected
  // orbit sums + total error, and record undo info for a possible reject().
  void apply_move_update() {
    changed_sites_.clear();
    for (std::size_t k = 0; k < occ_.size(); ++k) {
      if (current_occ(k) != occ_[k]) {
        changed_sites_.push_back(k);
      }
    }
    if (changed_sites_.empty()) {
      return; // no occupation change (e.g. generator found no candidate)
    }
    ++epoch_;
    affected_insts_.clear();
    affected_orbits_.clear();
    for (const std::size_t site : changed_sites_) {
      for (const std::uint32_t gid : site_to_instances_[site]) {
        if (seen_inst_[gid] != epoch_) {
          seen_inst_[gid] = epoch_;
          affected_insts_.push_back(gid);
        }
        const std::size_t o = inst_orbit_[gid];
        if (seen_orbit_[o] != epoch_) {
          seen_orbit_[o] = epoch_;
          affected_orbits_.push_back(o);
        }
      }
    }
    // Save undo (occupations of changed sites, raw sums of affected orbits).
    undo_sites_.clear();
    for (const std::size_t site : changed_sites_) {
      undo_sites_.emplace_back(site, occ_[site]);
    }
    undo_orbit_sums_.clear();
    for (const std::size_t o : affected_orbits_) {
      undo_orbit_sums_.emplace_back(o, orbit_sum_[o]);
    }
    // Old per-instance products (computed against the pre-move occ_).
    old_prod_.resize(affected_insts_.size());
    for (std::size_t a = 0; a < affected_insts_.size(); ++a) {
      old_prod_[a] = instance_product(affected_insts_[a]);
    }
    // Commit the new occupations, then apply the per-instance product deltas.
    for (const std::size_t site : changed_sites_) {
      occ_[site] = current_occ(site);
    }
    for (std::size_t a = 0; a < affected_insts_.size(); ++a) {
      const std::uint32_t gid = affected_insts_[a];
      orbit_sum_[inst_orbit_[gid]] += instance_product(gid) - old_prod_[a];
    }
    // Patch total_err_ for the affected orbits using their saved old sums.
    for (const auto &[o, old_sum] : undo_orbit_sums_) {
      if (orbit_count_[o] == 0) {
        continue;
      }
      const double cnt = static_cast<double>(orbit_count_[o]);
      const double old_dev = old_sum / cnt - orbits_[o].target;
      const double new_dev = orbit_sum_[o] / cnt - orbits_[o].target;
      total_err_ += orbits_[o].weight * (new_dev * new_dev - old_dev * old_dev);
    }
  }

  // Undo apply_move_update() on a rejected move (caller restores total_err_).
  void rollback_move() noexcept {
    for (const auto &[o, old_sum] : undo_orbit_sums_) {
      orbit_sum_[o] = old_sum;
    }
    for (const auto &[site, old_occ] : undo_sites_) {
      occ_[site] = old_occ;
    }
    undo_sites_.clear();
    undo_orbit_sums_.clear();
  }

  const AtomicStructure &structure_;
  std::unordered_map<std::string, int> occ_index_; // element → occupation index
  // Flat occupation-index → occupation-index table for the integer fast path in
  // refresh_site_occ(); empty ⇒ fall back to the occ_index_ string lookup.
  std::vector<int> occ_of_code_;
  CorrFuncTable table_;
  std::vector<ClusterOrbit> orbits_;
  // Per-site occupation index, refreshed once per evaluation from
  // structure_.elements; -1 = unknown species. Source of the hot-loop reads.
  // mutable: refreshed by const current_correlations() as well as compute_error.
  mutable std::vector<int> site_occ_;
  std::size_t total_instances_ = [this] {
    return std::transform_reduce(
        orbits_.begin(), orbits_.end(), std::size_t{0}, std::plus<>{},
        [](const auto &o) { return o.instance_count(); });
  }();

  // --- Incremental state (built lazily by ensure_built) --------------------
  bool built_ = false;
  std::vector<int> occ_;            // committed per-site occupation index
  std::vector<double> orbit_sum_;   // running raw Σ of per-instance products
  std::vector<std::size_t> orbit_count_; // cached instance_count per orbit
  double total_err_ = 0.0;          // running Σ weight·(corr−target)²
  std::vector<std::size_t> orbit_base_;  // global-instance-id base per orbit
  std::vector<std::uint32_t> inst_orbit_; // global instance id → orbit index
  std::vector<std::vector<std::uint32_t>> site_to_instances_; // site → gids
  // Per-step dedup of touched instances/orbits via an epoch stamp.
  std::uint64_t epoch_ = 0;
  std::vector<std::uint64_t> seen_inst_;
  std::vector<std::uint64_t> seen_orbit_;
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

static_assert(CConstraint<ClusterCorrelationConstraint>,
              "ClusterCorrelationConstraint must satisfy the CConstraint concept");

} // namespace RMC
