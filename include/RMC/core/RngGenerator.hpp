#pragma once
#include <array>
#include <boost/coroutine2/all.hpp>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>

namespace RMC {

// Batch buffer backed by a boost::coroutines2 generator.
//
// The coroutine owns its engine + distribution + fill buffer on its own
// heap-allocated stack. It default-constructs Dist (so uniform → [0,1),
// normal → N(0,1)) and yields a std::span over the freshly-filled array.
// The span is valid while the coroutine is suspended; the caller must not
// resume the coroutine while holding the span.
//
// Context-switch cost (~150 cycles, boost::context) is amortised over N
// values per yield — for N=1024 this is ~0.15 cycles overhead per sample.
//
// Copy semantics: copies restart from the same seed (independent fresh stream).
template <typename Dist, std::size_t N = 1024> class RngBatchBuffer {
  using T = typename Dist::result_type;
  using coro_t = boost::coroutines2::coroutine<std::span<const T>>;

  std::uint32_t seed_;
  typename coro_t::pull_type gen_;
  std::span<const T> batch_;
  std::size_t pos_{0};

  static auto make_gen(std::uint32_t s) {
    return typename coro_t::pull_type{[s](typename coro_t::push_type &sink) {
      std::mt19937 eng{s};
      Dist dist{};
      std::array<T, N> buf;
      while (true) {
        for (auto &v : buf)
          v = dist(eng);
        sink(std::span<const T>{buf});
      }
    }};
  }

public:
  explicit RngBatchBuffer(std::uint32_t seed)
      : seed_(seed), gen_(make_gen(seed)), batch_(gen_.get()) {}

  RngBatchBuffer(const RngBatchBuffer &o) : RngBatchBuffer(o.seed_) {}
  RngBatchBuffer &operator=(const RngBatchBuffer &o) {
    return *this = RngBatchBuffer(o.seed_);
  }
  RngBatchBuffer(RngBatchBuffer &&) = default;
  RngBatchBuffer &operator=(RngBatchBuffer &&) = default;

  T next() {
    if (pos_ == N) {
      gen_();
      batch_ = gen_.get();
      pos_ = 0;
    }
    return batch_[pos_++];
  }
};

// Convenience façade exposing three independent random streams:
//   uniform()        → [0, 1)   batched via coroutine
//   uniform(lo, hi)  → [lo, hi) scaled from the [0,1) buffer
//   normal()         → N(0, 1)  batched via coroutine
//   engine()         → raw std::mt19937 for ad-hoc distributions
//                      (uniform_int, discrete, etc.)
//
// Seeds are offset by 1/2 so the three streams are statistically independent.
// Copy semantics inherited from RngBatchBuffer: copies restart from same seed.
template <std::size_t N = 1024> class RngBuffer {
public:
  explicit RngBuffer(std::uint32_t seed = 42)
      : seed_(seed), uniform_buf_(seed), normal_buf_(seed + 1),
        engine_(seed + 2) {}

  RngBuffer(const RngBuffer &o) : RngBuffer(o.seed_) {}
  RngBuffer &operator=(const RngBuffer &o) {
    return *this = RngBuffer(o.seed_);
  }
  RngBuffer(RngBuffer &&) = default;
  RngBuffer &operator=(RngBuffer &&) = default;

  double uniform() { return uniform_buf_.next(); }
  double uniform(double lo, double hi) {
    return lo + (hi - lo) * uniform_buf_.next();
  }
  double normal() { return normal_buf_.next(); }

  std::mt19937 &engine() noexcept { return engine_; }

private:
  std::uint32_t seed_;
  RngBatchBuffer<std::uniform_real_distribution<double>, N> uniform_buf_;
  RngBatchBuffer<std::normal_distribution<double>, N> normal_buf_;
  std::mt19937 engine_;
};

} // namespace RMC
