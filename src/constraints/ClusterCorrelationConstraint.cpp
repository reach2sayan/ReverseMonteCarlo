#include <RMC/constraints/ClusterCorrelationConstraint.hpp>

#include <algorithm>
#include <numeric>
#include <ranges>

namespace RMC {

ClusterCorrelationConstraint::ClusterCorrelationConstraint(
    const AtomicStructure &structure, const SpeciesMap &species_map,
    std::vector<ClusterOrbit> orbits)
    : structure_(structure), orbits_(std::move(orbits)) {
  // Degenerate single-block table: one func, σ per occupation.
  table_.push_back(
      mat_t::Zero(1, static_cast<Eigen::Index>(species_map.size())));
  for (const auto [i, kv] : species_map | std::views::enumerate) {
    const auto &[elem, sigma] = kv;
    occ_index_[elem] = static_cast<int>(i);
    table_[0](0, i) = sigma;
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

double
ClusterCorrelationConstraint::compute_error(const coords_t &,
                                            std::span<const std::size_t>) {
  refresh_site_occ();
  return std::ranges::fold_left(
      orbits_, 0.0, [this](double total, const ClusterOrbit &orbit) {
        const double dev = orbit_correlation(orbit) - orbit.target;
        return total + orbit.weight * dev * dev;
      });
}

void ClusterCorrelationConstraint::compute_before_move(
    const coords_t &, std::span<const std::size_t>) {
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
    const coords_t &, std::span<const std::size_t>) {
  apply_move_update();
  err_after_ = total_err_;
}

void ClusterCorrelationConstraint::accept() noexcept {
  err_before_ = err_after_;
  undo_sites_.clear();
  undo_orbit_sums_.clear();
  if (++accepted_since_resync_ >= kResyncInterval) {
    resync_full(); // re-sync state vectors are pre-sized: no allocation
    accepted_since_resync_ = 0;
    err_before_ = err_after_ = total_err_;
  }
}

void ClusterCorrelationConstraint::reject() noexcept {
  rollback_move();
  total_err_ = err_before_;
  err_after_ = err_before_;
}

std::vector<double> ClusterCorrelationConstraint::current_correlations() const {
  refresh_site_occ();
  return orbits_ | std::views::transform([this](const auto &orbit) {
           return orbit_correlation(orbit);
         }) |
         std::ranges::to<std::vector>();
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

// Resolve every site's occupation index once per evaluation, so the hot orbit
// loop indexes a flat int array instead of hashing the element symbol per
// cluster point. -1 marks an unknown or removed species.
void ClusterCorrelationConstraint::refresh_site_occ() const {
  site_occ_.resize(structure_.elements.size());
  std::ranges::transform(std::views::iota(std::size_t{0}, site_occ_.size()),
                         site_occ_.begin(),
                         [this](std::size_t s) { return current_occ(s); });
}

double ClusterCorrelationConstraint::orbit_correlation(
    const ClusterOrbit &orbit) const {
  const std::size_t count = orbit.instance_count();
  if (count == 0) {
    return 0.0;
  }
  const double sum = std::ranges::fold_left(
      orbit.instances(), 0.0, [&](double acc, const auto &inst) {
        return acc + product(orbit, std::span<const std::size_t>(inst), site_occ_);
      });
  return sum / static_cast<double>(count);
}

// Occupation index of site k from the live structure: the integer fast path
// via atomic_numbers when they encode occupation indices, else the element
// symbol. -1 for a removed atom or an unknown species.
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

double ClusterCorrelationConstraint::product(const ClusterOrbit &orb,
                                             std::span<const std::size_t> sites,
                                             std::span<const int> occ) const {
  double prod = 1.0;
  for (const auto [p, site] : sites | std::views::enumerate) {
    const int o = occ[site];
    if (o < 0) {
      return 0.0; // unknown species → zero contribution (legacy behaviour)
    }
    const auto pp = static_cast<std::size_t>(p);
    prod *= table_[static_cast<std::size_t>(orb.site_type_at(pp))](
        orb.func_at(pp), o);
  }
  return prod;
}

// Product over one instance (global id) against the committed occ_.
double ClusterCorrelationConstraint::instance_product(std::uint32_t gid) const {
  return product(orbits_[index_.orbit_of(gid)], sites_of(gid), occ_);
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
    st.sum = std::ranges::fold_left(
        std::views::iota(std::size_t{0}, st.count), 0.0,
        [&](double acc, std::size_t li) {
          return acc + instance_product(static_cast<std::uint32_t>(st.base + li));
        });
    if (st.count == 0) {
      continue;
    }
    const double dev = st.sum / static_cast<double>(st.count) - orbits_[o].target;
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

void ClusterCorrelationConstraint::apply_move_update() {
  changed_sites_.clear();
  std::ranges::copy_if(std::views::iota(std::size_t{0}, occ_.size()),
                       std::back_inserter(changed_sites_), [&](std::size_t k) {
                         return current_occ(k) != occ_[k];
                       });
  if (changed_sites_.empty()) {
    return; // no occupation change (e.g. the generator found no candidate)
  }
  collect_affected();
  save_undo(); // against the still-committed occ_
  for (const std::size_t site : changed_sites_) {
    occ_[site] = current_occ(site);
  }
  apply_product_deltas();
  patch_orbit_errors();
}

// Gather the instances and orbits touched by the changed sites, de-duplicated
// within this step via the epoch stamp.
void ClusterCorrelationConstraint::collect_affected() {
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
}

// Record undo info (changed sites' old occ, affected orbits' old sums) and the
// pre-move per-instance products, all against the still-committed occ_.
void ClusterCorrelationConstraint::save_undo() {
  undo_sites_.clear();
  std::ranges::transform(
      changed_sites_, std::back_inserter(undo_sites_),
      [&](std::size_t site) { return std::pair{site, occ_[site]}; });

  undo_orbit_sums_.clear();
  std::ranges::transform(
      affected_orbits_, std::back_inserter(undo_orbit_sums_),
      [&](std::size_t o) { return std::pair{o, ostate_[o].sum}; });

  old_prod_.resize(affected_insts_.size());
  std::ranges::transform(affected_insts_, old_prod_.begin(),
                         [&](auto gid) { return instance_product(gid); });
}

// Fold each affected instance's product delta into its orbit's sum, now that
// occ_ holds the new occupations.
void ClusterCorrelationConstraint::apply_product_deltas() {
  for (const auto [gid, old_prod] :
       std::views::zip(affected_insts_, old_prod_)) {
    ostate_[index_.orbit_of(gid)].sum += instance_product(gid) - old_prod;
  }
}

// Patch total_err_ for the affected orbits using their saved old sums.
void ClusterCorrelationConstraint::patch_orbit_errors() {
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
