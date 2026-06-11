#pragma once
#include "RenderSnapshot.hpp"

#include <RMC/RMCRunner.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rmcgui {

enum class RunState { Idle, Building, Running, Paused, Finished, Error };

// Owns the move-only RMC::Engine on a background worker thread and is the ONLY
// thread that touches it. The GUI thread interacts solely through atomics
// (start/pause/stop) and the mutex-guarded snapshot/history/breakdown copies.
class SimController {
public:
  SimController() = default;
  ~SimController();
  SimController(const SimController &) = delete;
  SimController &operator=(const SimController &) = delete;

  // (Re)start a run from cfg. Joins any prior worker first.
  void start(const RMC::RMCConfig &cfg);
  void pause();
  void resume();
  void stop(); // cooperative; the worker exits within one chunk
  void shutdown(); // stop + join the worker (call before tearing down sinks)

  [[nodiscard]] RunState state() const { return state_.load(); }
  [[nodiscard]] bool active() const;

  // Latest render-ready snapshot (cheap O(N) copy); generation 0 ⇒ nothing yet.
  [[nodiscard]] RenderSnapshot snapshot() const;

  // χ² convergence history copied out under lock.
  void convergence(std::vector<float> &steps, std::vector<float> &chi2) const;

  // Per-constraint {name, error} at the latest logged step.
  [[nodiscard]] std::vector<std::pair<std::string, double>> breakdown() const;

  [[nodiscard]] std::string last_error() const;

private:
  void join_worker();
  void worker_main(RMC::RMCConfig cfg);
  void publish(RMC::Engine &eng, std::uint64_t step, std::uint64_t acc,
               std::uint64_t tried, double chi2, const RMC::AtomicStructure &s);
  void wait_while_paused();

  std::thread worker_;
  std::atomic<RunState> state_{RunState::Idle};
  std::atomic<bool> stop_req_{false};
  std::atomic<bool> pause_req_{false};

  std::mutex pause_mtx_;
  std::condition_variable pause_cv_;

  mutable std::mutex data_mtx_; // guards everything below
  RenderSnapshot latest_;
  std::array<float, 9> box_flat_{};
  std::vector<float> hist_step_, hist_chi2_;
  std::vector<std::pair<std::string, double>> breakdown_;
  std::string last_error_;
  std::uint64_t gen_counter_ = 0;
};

} // namespace rmcgui
