#include <RMC/Ensemble.hpp>

#include <algorithm>
#include <cstdlib>
#include <thread>

namespace RMC::detail {

std::size_t allocated_cpus() noexcept {
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

} // namespace RMC::detail
