#pragma once
#include <boost/random/normal_distribution.hpp>
#include <boost/random/taus88.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <cstddef>
#include <cstdint>

namespace RMC {

// A per-object random stream: one small-state engine (taus88, three words)
// with on-demand distributions, so every generator/selector owns one cheaply.
class Rng {
public:
  explicit Rng(std::uint32_t seed = 42) : engine_(seed) {}
  double uniform() { return boost::random::uniform_01<double>{}(engine_); }
  double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
  double normal() { return boost::random::normal_distribution<double>{}(engine_); }
  // Uniform index in [0, n).
  std::size_t index(std::size_t n) {
    return boost::random::uniform_int_distribution<std::size_t>{0, n - 1}(engine_);
  }
  boost::random::taus88 &engine() noexcept { return engine_; }

private:
  boost::random::taus88 engine_;
};

} // namespace RMC
