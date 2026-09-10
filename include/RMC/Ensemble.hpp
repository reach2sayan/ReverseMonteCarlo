#pragma once
#include <RMC/Engine.hpp>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <ranges>
#include <vector>

#if defined(RMC_USE_TBB)
#include <oneapi/tbb/global_control.h>
#endif

namespace RMC {

namespace detail {

template <class F>
std::vector<Engine> make_engines(F &make_engine, std::size_t n) {
  return std::views::iota(std::size_t{0}, n) |
         std::views::transform(std::ref(make_engine)) |
         std::ranges::to<std::vector>();
}

// Runs body(i) for every replica on its own thread and rethrows the first
// worker exception. With TBB each replica gets tbb_threads_per_replica workers
// (0 = auto: allocated_cpus() / n, at least 1).
template <class Body>
void run_replicas(std::size_t n,
                  [[maybe_unused]] std::size_t tbb_threads_per_replica,
                  const Body &body) {
#if defined(RMC_USE_TBB)
  const tbb::global_control gc(
      tbb::global_control::max_allowed_parallelism,
      tbb_threads_per_replica > 0
          ? tbb_threads_per_replica
          : std::max(std::size_t{1}, allocated_cpus() / n));
#endif
  auto workers = std::views::iota(std::size_t{0}, n) |
                 std::views::transform([&](std::size_t i) {
                   return std::async(std::launch::async, std::cref(body), i);
                 }) |
                 std::ranges::to<std::vector>();
  std::ranges::for_each(workers, [](auto &f) { f.get(); });
}

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
                    std::size_t tbb_threads_per_replica = 0,
                    Prepare prepare = {}) {
  auto engines = detail::make_engines(make_engine, n_replicas);
  std::ranges::for_each(engines, prepare);
  detail::run_replicas(n_replicas, tbb_threads_per_replica,
                       [&](std::size_t i) { engines[i].run(n_steps); });
  return std::move(*std::ranges::min_element(engines, {}, &Engine::best_error));
}

// Runs n_replicas engines in parallel. Every sync_every steps the best replica's
// structure is broadcast to all laggards (each keeps its own selector/RNG, so
// they re-diverge). Stops when any replica reaches target_chi2 (or max_steps);
// returns the replica with the lowest current chi2.
template <std::invocable<std::size_t> F>
  requires std::same_as<std::invoke_result_t<F, std::size_t>, Engine>
Engine run_ensemble_cooperative(F make_engine, std::size_t n_replicas,
                                double target_chi2,
                                std::uint64_t sync_every = 1000,
                                std::uint64_t max_steps = 0,
                                std::size_t tbb_threads_per_replica = 0) {
  auto engines = detail::make_engines(make_engine, n_replicas);
  const auto best = [&] {
    return std::ranges::min_element(
        engines, {}, [](const Engine &e) { return e.stats().last_total_err; });
  };

  // Broadcast only the mutable state (not the whole Engine): each laggard copies
  // it into its own engine, keeping its constraints/generators bound.
  AtomicStructure shared_best = engines[0].structure();
  std::atomic<bool> done{false};
  std::atomic<std::size_t> best_i{0};
  std::barrier sync_point(static_cast<std::ptrdiff_t>(n_replicas), [&]() noexcept {
    const auto b = best();
    done.store(b->stats().last_total_err <= target_chi2, std::memory_order_relaxed);
    best_i.store(static_cast<std::size_t>(b - engines.begin()),
                 std::memory_order_relaxed);
    shared_best.assign_mutable_state(b->structure());
  });

  detail::run_replicas(n_replicas, tbb_threads_per_replica, [&](std::size_t i) {
    for (std::uint64_t steps = 0;;) {
      const auto n = max_steps > 0 ? std::min(sync_every, max_steps - steps)
                                   : sync_every;
      engines[i].run(n);
      steps += n;
      sync_point.arrive_and_wait();
      if (done || (max_steps > 0 && steps >= max_steps)) {
        return;
      }
      if (i != best_i.load(std::memory_order_relaxed)) {
        engines[i].structure().assign_mutable_state(shared_best);
      }
    }
  });
  return std::move(*best());
}

} // namespace RMC
