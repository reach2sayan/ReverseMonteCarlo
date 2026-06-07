#pragma once
#include <RMC/Engine.hpp>

#include <boost/log/trivial.hpp>

#include <atomic>
#include <barrier>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <thread>
#include <vector>

#if defined(RMC_USE_TBB)
#include <oneapi/tbb/global_control.h>
#endif

#include <cstdlib>

namespace RMC {

namespace detail {

// Returns the number of CPUs allocated to this process, in priority order:
//   1. SLURM_CPUS_PER_TASK  (SLURM scheduler)
//   2. PBS_NUM_PPN           (PBS/Torque scheduler)
//   3. LSB_DJOB_NUMPROC      (LSF scheduler)
//   4. std::thread::hardware_concurrency() (local fallback)
inline std::size_t allocated_cpus() noexcept {
  for (const char *var :
       {"SLURM_CPUS_PER_TASK", "PBS_NUM_PPN", "LSB_DJOB_NUMPROC"}) {
    if (const char *val = std::getenv(var); val && *val) {
      if (const int n = std::atoi(val); n > 0) {
        return static_cast<std::size_t>(n);
      }
    }
  }
  return std::max(1u, std::thread::hardware_concurrency());
}

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

// Runs n_replicas engines in parallel threads (each built by make_engine(i))
// for n_steps each, then returns the replica with the lowest total chi2 error.
// make_engine(i) must return a fully configured Engine ready to run.
//
// The returned Engine is moved into an internal vector. This is safe even when
// its constraints / move generators hold references into engine.structure()
// (e.g. SQS), because Engine keeps its structure on the heap at a stable address
// (see Engine), so the move does not relocate it.
//
// All engines are constructed in the calling thread so that
// boost::context coroutine lifetimes stay on one thread.
// tbb_threads_per_replica: TBB workers per replica engine (RMC_USE_TBB only).
//   0 = auto (uses SLURM_CPUS_PER_TASK / PBS_NUM_PPN / hardware_concurrency,
//             divided by n_replicas).  Set explicitly when running inside a
//             scheduler that does not export those variables.
template <std::invocable<std::size_t> F>
  requires std::same_as<std::invoke_result_t<F, std::size_t>, Engine>
Engine run_ensemble(F make_engine, std::size_t n_replicas,
                    std::uint64_t n_steps,
                    [[maybe_unused]] std::size_t tbb_threads_per_replica = 0) {
  std::vector<Engine> engines;
  engines.reserve(n_replicas);
  for (std::size_t i = 0; i < n_replicas; ++i) {
    engines.push_back(make_engine(i));
  }

  auto run_fn = [&](std::size_t i) { engines[i].run(n_steps); };
  std::vector<std::packaged_task<void()>> tasks;
  std::vector<std::future<void>> futs;
  tasks.reserve(n_replicas);
  futs.reserve(n_replicas);
  for (std::size_t i = 0; i < n_replicas; ++i) {
    tasks.emplace_back([&run_fn, i] { run_fn(i); });
    futs.push_back(tasks.back().get_future());
  }
#if defined(RMC_USE_TBB)
  tbb::global_control tbb_gc(
      tbb::global_control::max_allowed_parallelism,
      detail::tbb_budget(n_replicas, tbb_threads_per_replica));
#endif
  std::vector<std::jthread> threads;
  threads.reserve(n_replicas);
  for (auto &task : tasks) {
    threads.emplace_back(std::move(task));
  }
  std::ranges::for_each(futs.begin(), futs.end(), [](auto &f) { f.get(); });

  std::size_t best_i = 0;
  double best_chi2 = engines[0].best_error();
  for (std::size_t i = 1; i < n_replicas; ++i) {
    double chi2 = engines[i].best_error();
    if (chi2 < best_chi2) {
      best_chi2 = chi2;
      best_i = i;
    }
  }
  return std::move(engines[best_i]);
}

// Runs n_replicas engines in parallel. Every sync_every steps all workers
// synchronise: the best (lowest chi2) replica's STRUCTURE is broadcast to all
// laggards, who continue the search from it (each keeps its own selector/RNG, so
// they re-diverge). Stops as soon as any replica reaches target_chi2 (or
// max_steps is exhausted). Returns the best engine seen.
template <std::invocable<std::size_t> F>
  requires std::same_as<std::invoke_result_t<F, std::size_t>, Engine>
Engine run_ensemble_cooperative(
    F make_engine, std::size_t n_replicas, double target_chi2,
    std::uint64_t sync_every = 1000, std::uint64_t max_steps = 0,
    [[maybe_unused]] std::size_t tbb_threads_per_replica = 0) {
  std::vector<Engine> engines;
  engines.reserve(n_replicas);
  for (std::size_t i = 0; i < n_replicas; ++i) {
    engines.push_back(make_engine(i));
  }

  // Broadcasting only the structure (not the whole Engine) keeps this move-only:
  // each laggard copies the best structure into its OWN engine, whose structure
  // lives at a stable address, so its constraints/generators stay bound.
  AtomicStructure shared_best_structure = engines[0].structure();
  std::atomic<bool> any_done{false};
  std::atomic<std::size_t> best_i_atomic{0};

  std::barrier sync_point{
      static_cast<std::ptrdiff_t>(n_replicas), [&]() noexcept {
        try {
          double best_chi2 = std::numeric_limits<double>::max();
          std::size_t best_i = 0;
          for (std::size_t j = 0; j < n_replicas; ++j) {
            double chi2 = engines[j].stats().last_total_err;
            if (chi2 < best_chi2) {
              best_chi2 = chi2;
              best_i = j;
            }
            if (chi2 <= target_chi2) {
              any_done.store(true, std::memory_order_relaxed);
            }
          }
          best_i_atomic.store(best_i, std::memory_order_relaxed);
          shared_best_structure = engines[best_i].structure();
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
        engines[i].structure() = shared_best_structure;
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
