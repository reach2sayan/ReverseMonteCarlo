#pragma once
#include <RMC/Engine.hpp>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <future>
#include <ranges>
#include <thread>
#include <vector>

#if defined(RMC_USE_TBB)
#include <oneapi/tbb/global_control.h>
#endif

#include <cstdlib>

namespace RMC {

namespace detail {

#if defined(RMC_USE_TBB)
// Compute per-replica TBB thread budget: explicit override > env heuristic.
// tbb_threads_per_replica == 0  →  auto (allocated_cpus / n_replicas, ≥ 1)
inline std::size_t tbb_budget(std::size_t n_replicas,
                              std::size_t tbb_threads_per_replica) noexcept {
  if (tbb_threads_per_replica > 0) {
    return tbb_threads_per_replica;
  }
  return std::max(std::size_t{1}, allocated_cpus() / n_replicas);
}
#endif

} // namespace detail

// Runs n_replicas engines (each built by make_engine(i)) for n_steps each in
// parallel threads, returns the lowest-chi2 replica. Engines keep their
// structure on the heap, so the move into the internal vector is safe even when
// constraints/generators hold references into engine.structure().
//
// tbb_threads_per_replica: TBB workers per replica (RMC_USE_TBB only). 0 = auto
//   (SLURM_CPUS_PER_TASK / PBS_NUM_PPN / hardware_concurrency, divided by
//   n_replicas); set explicitly under a scheduler that does not export those.
// prepare(engine) runs on each replica after it lands at its final address,
// before it runs — for setup that must bind to that address (e.g. gradient
// generators holding a pointer into engine.constraints()).
struct NoPrepare {
  void operator()(Engine &) const noexcept {}
};

template <std::invocable<std::size_t> F,
          std::invocable<Engine &> Prepare = NoPrepare>
  requires std::same_as<std::invoke_result_t<F, std::size_t>, Engine>
Engine run_ensemble(F make_engine, std::size_t n_replicas,
                    std::uint64_t n_steps,
                    [[maybe_unused]] std::size_t tbb_threads_per_replica = 0,
                    Prepare prepare = {}) {
  std::vector<Engine> engines;
  engines.reserve(n_replicas);

  std::size_t i = 0;
  std::ranges::generate_n(std::back_inserter(engines), n_replicas,
                          [&] { return make_engine(i++); });

  for (Engine &e : engines) {
    prepare(e);
  }

  auto run_fn = [&](std::size_t i) { engines[i].run(n_steps); };

  std::vector<std::packaged_task<void()>> tasks;
  std::vector<std::future<void>> futs;
  tasks.reserve(n_replicas);
  futs.reserve(n_replicas);
  for (std::size_t j = 0; j < n_replicas; ++j) {
    tasks.emplace_back([&run_fn, j] { run_fn(j); });
    futs.push_back(tasks.back().get_future());
  }

#if defined(RMC_USE_TBB)
  tbb::global_control tbb_gc(
      tbb::global_control::max_allowed_parallelism,
      detail::tbb_budget(n_replicas, tbb_threads_per_replica));
#endif

  std::vector<std::jthread> threads;
  threads.reserve(n_replicas);
  std::ranges::transform(tasks, std::back_inserter(threads), [](auto &task) {
    return std::jthread(std::move(task));
  });

  std::ranges::for_each(futs.begin(), futs.end(), [](auto &f) { f.get(); });

  auto best_it = std::ranges::min_element(engines, {}, &Engine::best_error);
  const auto best_i =
      static_cast<std::size_t>(std::ranges::distance(engines.begin(), best_it));

  return std::move(engines[best_i]);
}

// Runs n_replicas engines in parallel. Every sync_every steps the best replica's
// structure is broadcast to all laggards (each keeps its own selector/RNG, so
// they re-diverge). Stops when any replica reaches target_chi2 (or max_steps).
template <std::invocable<std::size_t> F>
  requires std::same_as<std::invoke_result_t<F, std::size_t>, Engine>
Engine run_ensemble_cooperative(
    F make_engine, std::size_t n_replicas, double target_chi2,
    std::uint64_t sync_every = 1000, std::uint64_t max_steps = 0,
    [[maybe_unused]] std::size_t tbb_threads_per_replica = 0) {

  std::vector<Engine> engines;
  engines.reserve(n_replicas);
  std::ranges::transform(std::views::iota(std::size_t{0}, n_replicas),
                         std::back_inserter(engines),
                         [&](std::size_t i) { return make_engine(i); });

  // Broadcast only the structure (not the whole Engine): each laggard copies it
  // into its own engine, keeping its constraints/generators bound.
  AtomicStructure shared_best_structure = engines[0].structure();
  std::atomic<bool> any_done{false};
  std::atomic<std::size_t> best_i_atomic{0};

  std::barrier sync_point{
      static_cast<std::ptrdiff_t>(n_replicas), [&]() noexcept {
        try {
          const auto chi2_of = [](const auto &e) {
            return e.stats().last_total_err;
          };
          const auto best = std::ranges::min_element(engines, {}, chi2_of);
          const auto best_i =
              static_cast<std::size_t>(best - engines.begin());
          // The minimum is <= target iff some replica reached the target.
          const bool done = chi2_of(*best) <= target_chi2;

          if (done) {
            any_done.store(true, std::memory_order_relaxed);
          }

          best_i_atomic.store(best_i, std::memory_order_relaxed);
          // Only the mutable fields differ between replicas; the label vectors
          // are identical, so skip deep-copying them every sync.
          shared_best_structure.assign_mutable_state(
              engines[best_i].structure());
        } catch (...) {
          std::terminate();
        }
      }};

  auto worker_fn = [&](std::size_t i) {
    std::uint64_t steps_done = 0;
    while (true) {
      auto n = (max_steps > 0) ? std::min(sync_every, max_steps - steps_done)
                               : sync_every;
      engines[i].run(n);
      steps_done += n;
      sync_point.arrive_and_wait();
      if (any_done || (max_steps > 0 && steps_done >= max_steps)) {
        return;
      }
      if (i != best_i_atomic.load(std::memory_order_relaxed)) {
        engines[i].structure().assign_mutable_state(shared_best_structure);
      }
    }
  };

  std::vector<std::packaged_task<void()>> tasks;
  std::vector<std::future<void>> futs;
  tasks.reserve(n_replicas);
  futs.reserve(n_replicas);
  for (std::size_t i = 0; i < n_replicas; ++i) {
    tasks.emplace_back([&worker_fn, i] { std::invoke(worker_fn, i); });
    futs.push_back(tasks.back().get_future());
  }

#if defined(RMC_USE_TBB)
  tbb::global_control tbb_gc_coop(
      tbb::global_control::max_allowed_parallelism,
      detail::tbb_budget(n_replicas, tbb_threads_per_replica));
#endif
  std::vector<std::jthread> threads;
  threads.reserve(n_replicas);
  for (auto &task : tasks) {
    threads.emplace_back(std::move(task));
  }
  std::ranges::for_each(
      futs, [](auto &f) { f.get(); }); // re-throws any worker exception

  std::size_t best_i = 0;
  double best_chi2 = engines[0].best_error();
  for (std::size_t i = 1; i < n_replicas; ++i) {
    const double chi2 = engines[i].stats().last_total_err;
    if (chi2 < best_chi2) {
      best_chi2 = chi2;
      best_i = i;
    }
  }
  return std::move(engines[best_i]);
}

} // namespace RMC
