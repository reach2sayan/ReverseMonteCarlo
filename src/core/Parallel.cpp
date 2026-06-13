#include <RMC/core/Parallel.hpp>

#if defined(RMC_USE_TBB)
#include <cstdlib>

namespace RMC::parallel {

tbb::task_arena &arena() {
  static tbb::task_arena a = [] {
    int max_threads = default_concurrency();
    if (const auto env = read_env("RMC_NUM_THREADS")) {
      if (const int n = std::atoi(env->c_str()); n > 0) {
        max_threads = n;
      }
    }
    return tbb::task_arena{max_threads};
  }();
  return a;
}

} // namespace RMC::parallel
#endif
