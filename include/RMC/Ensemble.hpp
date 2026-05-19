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

namespace RMC {

// Runs n_replicas engines in parallel threads (each built by make_engine(i))
// for n_steps each, then returns the replica with the lowest total chi2 error.
// make_engine(i) must return a fully configured Engine ready to run.
//
// All engines are constructed in the calling thread so that
// boost::context coroutine lifetimes stay on one thread.
template <std::invocable<std::size_t> F>
  requires std::same_as<std::invoke_result_t<F, std::size_t>, Engine>
Engine run_ensemble(F make_engine, std::size_t n_replicas,
                    std::uint64_t n_steps) {
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
  std::vector<std::jthread> threads;
  threads.reserve(n_replicas);
  for (auto &task : tasks) {
    threads.emplace_back(std::move(task));
  }
  std::ranges::for_each(futs.begin(), futs.end(), [](auto &f) { f.get(); });

  std::size_t best_i = 0;
  double best_chi2 = engines[0].stats().last_total_err;
  for (std::size_t i = 1; i < n_replicas; ++i) {
    double chi2 = engines[i].stats().last_total_err;
    if (chi2 < best_chi2) {
      best_chi2 = chi2;
      best_i = i;
    }
  }
  return std::move(engines[best_i]);
}

// Runs n_replicas engines in parallel. Every sync_every steps all workers
// synchronise: the best (lowest chi2) engine is broadcast to all laggards,
// then all continue from that state. Stops as soon as any replica reaches
// target_chi2 (or max_steps is exhausted). Returns the best engine seen.
template <std::invocable<std::size_t> F>
  requires std::same_as<std::invoke_result_t<F, std::size_t>, Engine>
Engine run_ensemble_cooperative(F make_engine, std::size_t n_replicas,
                                double target_chi2,
                                std::uint64_t sync_every = 1000,
                                std::uint64_t max_steps = 0) {
  std::vector<Engine> engines;
  engines.reserve(n_replicas);
  for (std::size_t i = 0; i < n_replicas; ++i) {
    engines.push_back(make_engine(i));
  }

  Engine shared_best = engines[0];
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
          shared_best = engines[best_i];
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
        engines[i] = shared_best;
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

  std::vector<std::jthread> threads;
  threads.reserve(n_replicas);
  for (auto &task : tasks) {
    threads.emplace_back(std::move(task));
  }
  std::ranges::for_each(
      futs, [](auto &f) { f.get(); }); // re-throws any worker exception
  return shared_best;
}

} // namespace RMC
