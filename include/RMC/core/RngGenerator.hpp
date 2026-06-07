#pragma once
#include <array>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cstddef>
#include <cstdint>

namespace RMC {

template <typename Dist, std::size_t N = 1024> class RngBatchBuffer {
  using T = typename Dist::result_type;

  boost::random::mt19937 eng_;
  Dist dist_{};
  std::array<T, N> buf_;
  std::size_t pos_{N}; // start "empty" -> refill on first next()

public:
  explicit RngBatchBuffer(std::uint32_t seed) : eng_(seed) {}
  T next() {
    if (pos_ == N) {
      for (auto &v : buf_) {
        v = dist_(eng_);
      }
      pos_ = 0;
    }
    return buf_[pos_++];
  }
};

template <std::size_t N = 1024> class RngBuffer {
public:
  explicit RngBuffer(std::uint32_t seed = 42)
      : uniform_buf_(seed), normal_buf_(seed + 1), engine_(seed + 2) {}
  double uniform() { return uniform_buf_.next(); }
  double uniform(double lo, double hi) {
    return lo + (hi - lo) * uniform_buf_.next();
  }
  double normal() { return normal_buf_.next(); }
  boost::random::mt19937 &engine() noexcept { return engine_; }

private:
  RngBatchBuffer<boost::random::uniform_real_distribution<double>, N>
      uniform_buf_;
  RngBatchBuffer<boost::random::normal_distribution<double>, N> normal_buf_;
  boost::random::mt19937 engine_;
};

} // namespace RMC
