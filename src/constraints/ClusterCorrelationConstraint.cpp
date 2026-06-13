#include <RMC/constraints/ClusterCorrelationConstraint.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ranges>
#include <numbers>

namespace RMC {

CorrFuncTable CorrFuncTable::trigonometric(int max_components) {
  CorrFuncTable tab;
  if (max_components < 2) {
    max_components = 2;
  }
  tab.t.resize(static_cast<std::size_t>(max_components - 1));
  for (int m = 2; m <= max_components; ++m) {
    auto &site = tab.t[static_cast<std::size_t>(m - 2)];
    site.resize(static_cast<std::size_t>(m - 1));
    for (int f = 0; f < m - 1; ++f) {
      site[static_cast<std::size_t>(f)].assign(static_cast<std::size_t>(m), 0.0);
    }
    for (int s = 0; s < m; ++s) {
      for (int k = 1; k <= m / 2; ++k) {
        site[static_cast<std::size_t>(2 * k - 2)][static_cast<std::size_t>(s)] =
            -std::cos(2.0 * std::numbers::pi * s * k / m);
      }
      for (int k = 1; k <= (m + 1) / 2 - 1; ++k) {
        site[static_cast<std::size_t>(2 * k - 1)][static_cast<std::size_t>(s)] =
            -std::sin(2.0 * std::numbers::pi * s * k / m);
      }
    }
  }
  return tab;
}

ClusterCorrelationConstraint::ClusterCorrelationConstraint(
    const AtomicStructure &structure, const SpeciesMap &species_map,
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

ClusterCorrelationConstraint::ClusterCorrelationConstraint(
    const AtomicStructure &structure,
    std::unordered_map<std::string, int> occ_index, CorrFuncTable table,
    std::vector<ClusterOrbit> orbits)
    : structure_(structure), occ_index_(std::move(occ_index)),
      table_(std::move(table)), orbits_(std::move(orbits)) {
  build_occ_of_code();
}

double ClusterCorrelationConstraint::compute_error(
    const coords_t &, std::span<const std::size_t>) {
  refresh_site_occ();
  return std::ranges::fold_left(
      orbits_, 0.0, [this](double total, const ClusterOrbit &orbit) {
        const double dev = orbit_correlation(orbit) - orbit.target;
        return total + orbit.weight * dev * dev;
      });
}

void ClusterCorrelationConstraint::compute_before_move(
    Constraint::Token, const coords_t &, std::span<const std::size_t>) {
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

void ClusterCorrelationConstraint::compute_after_move(
    Constraint::Token, const coords_t &, std::span<const std::size_t>) {
  apply_move_update();
  err_after_ = total_err_;
}

void ClusterCorrelationConstraint::accept(Constraint::Token) noexcept {
  err_before_ = err_after_;
  undo_sites_.clear();
  undo_orbit_sums_.clear();
  if (++accepted_since_resync_ >= kResyncInterval) {
    resync_full(); // re-sync state vectors are pre-sized: no allocation
    accepted_since_resync_ = 0;
    err_before_ = err_after_ = total_err_;
  }
}

void ClusterCorrelationConstraint::reject(Constraint::Token) noexcept {
  rollback_move();
  total_err_ = err_before_;
  err_after_ = err_before_;
}

std::vector<double> ClusterCorrelationConstraint::current_correlations() const {
  refresh_site_occ();
  auto corr = orbits_ | std::views::transform([this](const auto &orbit) {
                return orbit_correlation(orbit);
              });
  return {corr.begin(), corr.end()};
}

// Build the flat occupation-index → occupation-index table used by the hot
// path. In the multicomponent (ATAT) setup structure_.atomic_numbers already
// stores the occupation index per site (ClusterEnumerator sets
// atomic_numbers[a] = occ_index[elements[a]]), so the table is the identity
// over the valid occupation indices with -1 in any gap. Empty table ⇒ the
// string fallback in refresh_site_occ() is used (binary / test path, where
// atomic_numbers carry true Z rather than occupation indices).
void ClusterCorrelationConstraint::build_occ_of_code() {
  if (occ_index_.empty()) {
    return;
  }
  const int max_occ = std::ranges::max(occ_index_ | std::views::values);
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
void ClusterCorrelationConstraint::refresh_site_occ() const {
  const std::size_t n = structure_.elements.size();
  site_occ_.resize(n);
  if (!occ_of_code_.empty()) {
    const int cap = static_cast<int>(occ_of_code_.size());
    std::ranges::transform(
        std::views::iota(std::size_t{0}, n), site_occ_.begin(),
        [&](std::size_t s) {
          const int code =
              structure_.atomic_numbers[static_cast<Eigen::Index>(s)];
          return (!absent(s) && code >= 0 && code < cap)
                     ? occ_of_code_[static_cast<std::size_t>(code)]
                     : -1;
        });
    return;
  }
  std::ranges::transform(std::views::iota(std::size_t{0}, n), site_occ_.begin(),
                         [&](std::size_t s) {
                           const auto it =
                               occ_index_.find(structure_.elements[s]);
                           return (absent(s) || it == occ_index_.end())
                                      ? -1
                                      : it->second;
                         });
}

double ClusterCorrelationConstraint::orbit_correlation(
    const ClusterOrbit &orbit) const {
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

// Current occupation index of site k from the live structure (mirrors the two
// paths in refresh_site_occ but for a single site).
int ClusterCorrelationConstraint::current_occ(std::size_t k) const {
  if (absent(k)) {
    return -1; // removed atom: drop every cluster instance that touches it
  }
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
double ClusterCorrelationConstraint::instance_product(std::uint32_t gid) const {
  const ClusterOrbit &orb = orbits_[index_.orbit_of(gid)];
  double prod = 1.0;
  for (const auto [p, site] : std::views::enumerate(sites_of(gid))) {
    const int occ = occ_[site];
    if (occ < 0) {
      return 0.0;
    }
    const int st = orb.site_types.empty() ? 0 : orb.site_types[p];
    const int fn = orb.funcs.empty() ? 0 : orb.funcs[p];
    prod *= table_.value(st, fn, occ);
  }
  return prod;
}

// The single place that maps a global instance id to its slice of flat_sites.
std::span<const std::size_t>
ClusterCorrelationConstraint::sites_of(std::uint32_t gid) const {
  const std::size_t o = index_.orbit_of(gid);
  const ClusterOrbit &orb = orbits_[o];
  const std::size_t off =
      (static_cast<std::size_t>(gid) - ostate_[o].base) * orb.body;
  return {orb.flat_sites.data() + off, orb.body};
}

// Build the site→instance reverse index and per-orbit globals (once).
void ClusterCorrelationConstraint::build_index() {
  const std::size_t n =
      std::max(structure_.elements.size(),
               static_cast<std::size_t>(structure_.atomic_numbers.size()));
  ostate_.assign(orbits_.size(), {});

  std::size_t base = 0;
  for (std::size_t o = 0; o < orbits_.size(); ++o) {
    ostate_[o].base = base;
    ostate_[o].count = orbits_[o].instance_count();
    base += ostate_[o].count;
  }
  const std::size_t total = base;

  index_.inst_orbit.assign(total, 0);
  index_.site_to_instances.assign(n, {});
  seen_inst_.assign(total, 0);

  for (std::size_t o = 0; o < orbits_.size(); ++o) {
    const ClusterOrbit &orb = orbits_[o];
    for (std::size_t li = 0; li < ostate_[o].count; ++li) {
      const auto gid = static_cast<std::uint32_t>(ostate_[o].base + li);
      index_.inst_orbit[gid] = static_cast<std::uint32_t>(o);
      const std::size_t off = li * orb.body;
      for (std::size_t p = 0; p < orb.body; ++p) {
        const std::size_t site = orb.flat_sites[off + p];
        if (site < index_.site_to_instances.size()) {
          index_.site_to_instances[site].push_back(gid);
        }
      }
    }
  }
}

// Recompute occ_, per-orbit sums and total_err_ from scratch (init + resync).
void ClusterCorrelationConstraint::resync_full() {
  const std::size_t n = index_.site_count();
  occ_.resize(n);
  std::ranges::transform(std::views::iota(std::size_t{0}, n), occ_.begin(),
                         [&](std::size_t k) { return current_occ(k); });
  total_err_ = 0.0;
  for (std::size_t o = 0; o < orbits_.size(); ++o) {
    OrbitState &st = ostate_[o];
    double sum = 0.0;
    for (std::size_t li = 0; li < st.count; ++li) {
      sum += instance_product(static_cast<std::uint32_t>(st.base + li));
    }
    st.sum = sum;
    if (st.count == 0) {
      continue;
    }
    const double dev = sum / static_cast<double>(st.count) - orbits_[o].target;
    total_err_ += orbits_[o].weight * dev * dev;
  }
}

void ClusterCorrelationConstraint::ensure_built() {
  if (built_) {
    return;
  }
  build_index();
  resync_full();
  built_ = true;
}

// True if the live structure no longer matches the committed occ_ baseline.
constexpr bool ClusterCorrelationConstraint::occ_mismatch() const {
  return std::ranges::any_of(
      std::views::iota(std::size_t{0}, occ_.size()),
      [&](std::size_t k) { return current_occ(k) != occ_[k]; });
}

// Apply the just-proposed move: diff changed sites, update only the affected
// orbit sums + total error, and record undo info for a possible reject().
void ClusterCorrelationConstraint::apply_move_update() {
  changed_sites_.clear();
  changed_sites_.reserve(occ_.size());
  std::ranges::copy_if(std::views::iota(std::size_t{0}, occ_.size()),
                       std::back_inserter(changed_sites_), [&](std::size_t k) {
                         return current_occ(k) != occ_[k];
                       });

  if (changed_sites_.empty()) {
    return; // no occupation change (e.g. generator found no candidate)
  }

  ++epoch_;
  affected_insts_.clear();
  affected_orbits_.clear();
  for (const std::size_t site : changed_sites_) {
    for (const std::uint32_t gid : index_.instances_of_site(site)) {
      if (seen_inst_[gid] != epoch_) {
        seen_inst_[gid] = epoch_;
        affected_insts_.push_back(gid);
      }
      const std::size_t o = index_.orbit_of(gid);
      if (ostate_[o].seen != epoch_) {
        ostate_[o].seen = epoch_;
        affected_orbits_.push_back(o);
      }
    }
  }
  // Save undo (occupations of changed sites, raw sums of affected orbits).
  undo_sites_.clear();
  undo_sites_.reserve(changed_sites_.size());
  std::ranges::transform(
      changed_sites_, std::back_inserter(undo_sites_),
      [&](std::size_t site) { return std::pair{site, occ_[site]}; });

  undo_orbit_sums_.clear();
  undo_orbit_sums_.reserve(affected_orbits_.size());
  std::ranges::transform(
      affected_orbits_, std::back_inserter(undo_orbit_sums_),
      [&](std::size_t o) { return std::pair{o, ostate_[o].sum}; });

  // Old per-instance products (computed against the pre-move occ_).
  old_prod_.resize(affected_insts_.size());
  std::ranges::transform(affected_insts_, old_prod_.begin(),
                         [&](auto gid) { return instance_product(gid); });

  // Commit the new occupations, then apply the per-instance product deltas.
  for (const std::size_t site : changed_sites_) {
    occ_[site] = current_occ(site);
  }

  for (const auto [gid, old_prod] : std::views::zip(affected_insts_, old_prod_)) {
    ostate_[index_.orbit_of(gid)].sum += instance_product(gid) - old_prod;
  }
  // Patch total_err_ for the affected orbits using their saved old sums.
  for (const auto &[o, old_sum] : undo_orbit_sums_) {
    const OrbitState &st = ostate_[o];
    if (st.count == 0) {
      continue;
    }
    const double cnt = static_cast<double>(st.count);
    const double old_dev = old_sum / cnt - orbits_[o].target;
    const double new_dev = st.sum / cnt - orbits_[o].target;
    total_err_ += orbits_[o].weight * (new_dev * new_dev - old_dev * old_dev);
  }
}

// Undo apply_move_update() on a rejected move (caller restores total_err_).
void ClusterCorrelationConstraint::rollback_move() noexcept {
  for (const auto &[o, old_sum] : undo_orbit_sums_) {
    ostate_[o].sum = old_sum;
  }
  for (const auto &[site, old_occ] : undo_sites_) {
    occ_[site] = old_occ;
  }
  undo_sites_.clear();
  undo_orbit_sums_.clear();
}

} // namespace RMC
