#pragma once
#include <RMC/core/Types.hpp>
#include <algorithm>
#include <utility>

#if defined(RMC_USE_TBB)
#include <execution>
#include <thread>

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

template <class It, class F> void for_each(It first, It last, F &&f) {
  arena().execute([&] {
    std::for_each(std::execution::par_unseq, first, last, std::forward<F>(f));
  });
}

template <class F>
void for_each(const std::ranges::input_range auto &&range, F &&f) {
  arena().execute([&] {
    std::for_each(std::execution::par_unseq, std::forward<F>(range).begin(),
                  std::forward<F>(range).end(), std::forward<F>(f));
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
