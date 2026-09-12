#pragma once
#include <RMC/core/Types.hpp>
#include <algorithm>
#include <utility>
#include <vector>

#if defined(RMC_USE_TBB)
#include <thread>

#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/task_arena.h>

namespace RMC::parallel {

inline int default_concurrency() noexcept {
  const unsigned hw = allocated_cpus();
  return hw > 0 ? static_cast<int>(hw) : 1;
}

// Process-wide arena; concurrency = all hardware threads or $RMC_NUM_THREADS.
tbb::task_arena &arena();
inline void set_max_concurrency(int n) {
  arena().terminate();
  arena().initialize(n);
}

// tbb::parallel_for_each, not std::for_each(par_unseq): the standard algorithm
// only reaches this arena on libstdc++, where par_unseq lowers to TBB anyway.
// MSVC's STL runs it on the Windows thread pool instead, which would ignore the
// arena -- and with it RMC_NUM_THREADS and Ensemble's per-replica split. The
// body is taken by value and invoked as f(*it), which is what every call site
// already expects.
template <class It, class F> void for_each(It first, It last, F &&f) {
  arena().execute(
      [&] { tbb::parallel_for_each(first, last, std::forward<F>(f)); });
}

} // namespace RMC::parallel

#else // !RMC_USE_TBB — serial fallbacks

namespace RMC::parallel {

constexpr FORCE_INLINE int default_concurrency() noexcept { return 1; }
constexpr FORCE_INLINE void set_max_concurrency(int) noexcept {}

template <class It, class F>
constexpr FORCE_INLINE void for_each(It first, It last, F &&f) {
  std::for_each(first, last, std::forward<F>(f));
}

} // namespace RMC::parallel

#endif

namespace RMC::parallel {

// Σ over i in [0, n) of body(i, acc), with one accumulator per lane: indices
// are dealt round-robin to default_concurrency() lanes (balancing triangular
// loops), each lane folds into its own copy of `zero`, and the lanes are
// summed at the end. Serial builds use a single lane.
template <class Acc, class Body>
Acc parallel_sum(std::size_t n, const Acc &zero, Body body) {
  const auto lanes = std::clamp<std::size_t>(
      static_cast<std::size_t>(default_concurrency()), 1, std::max<std::size_t>(n, 1));
  std::vector<Acc> acc(lanes, zero);
  const auto ids =
      std::views::iota(std::size_t{0}, lanes) | std::ranges::to<std::vector>();
  RMC::parallel::for_each(ids.begin(), ids.end(), [&](std::size_t lane) {
    for (std::size_t i = lane; i < n; i += lanes) {
      body(i, acc[lane]);
    }
  });
  for (std::size_t lane = 1; lane < lanes; ++lane) {
    acc[0] += acc[lane];
  }
  return std::move(acc[0]);
}

} // namespace RMC::parallel
