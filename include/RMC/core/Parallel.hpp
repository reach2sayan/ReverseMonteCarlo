#pragma once

#include <algorithm>
#include <utility>

#if defined(RMC_USE_TBB)
#include <cstdlib>
#include <execution>
#include <thread>

#include <oneapi/tbb/task_arena.h>

namespace RMC::parallel {

inline int default_concurrency() noexcept {
  const unsigned hw = std::thread::hardware_concurrency();
  return hw > 0 ? static_cast<int>(hw) : 1;
}

// Process-wide arena. Concurrency defaults to all hardware threads, or to
// $RMC_NUM_THREADS when that env var is set to a positive int.
inline tbb::task_arena &arena() {
  static tbb::task_arena a = [] {
    int max_threads = default_concurrency();
    if (const char *env = std::getenv("RMC_NUM_THREADS")) {
      if (const int n = std::atoi(env); n > 0) {
        max_threads = n;
      }
    }
    return tbb::task_arena{max_threads};
  }();
  return a;
}

inline void set_max_concurrency(int n) {
  arena().terminate();
  arena().initialize(n);
}

template <class It, class F> void for_each(It first, It last, F &&f) {
  arena().execute([&] {
    std::for_each(std::execution::par_unseq, first, last, std::forward<F>(f));
  });
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
