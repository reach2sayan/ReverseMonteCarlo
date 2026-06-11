#include "SimController.hpp"

#include <boost/leaf.hpp>

#include <algorithm>
#include <optional>

namespace leaf = boost::leaf;

namespace rmcgui {

SimController::~SimController() { join_worker(); }

bool SimController::active() const {
  const RunState s = state_.load();
  return s == RunState::Building || s == RunState::Running ||
         s == RunState::Paused;
}

void SimController::join_worker() {
  stop_req_.store(true);
  pause_req_.store(false);
  pause_cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void SimController::start(const RMC::RMCConfig &cfg) {
  join_worker();
  stop_req_.store(false);
  pause_req_.store(false);
  {
    std::scoped_lock lk(data_mtx_);
    latest_ = RenderSnapshot{};
    box_flat_ = {};
    hist_step_.clear();
    hist_chi2_.clear();
    breakdown_.clear();
    last_error_.clear();
    gen_counter_ = 0;
  }
  state_.store(RunState::Building);
  worker_ = std::thread([this, cfg] { worker_main(cfg); });
}

void SimController::pause() {
  if (state_.load() == RunState::Running) {
    pause_req_.store(true);
  }
}

void SimController::resume() {
  pause_req_.store(false);
  pause_cv_.notify_all();
}

void SimController::stop() {
  stop_req_.store(true);
  pause_req_.store(false);
  pause_cv_.notify_all();
}

void SimController::shutdown() { join_worker(); }

void SimController::wait_while_paused() {
  std::unique_lock lk(pause_mtx_);
  pause_cv_.wait(lk,
                 [this] { return !pause_req_.load() || stop_req_.load(); });
}

RenderSnapshot SimController::snapshot() const {
  std::scoped_lock lk(data_mtx_);
  return latest_;
}

void SimController::convergence(std::vector<float> &steps,
                                std::vector<float> &chi2) const {
  std::scoped_lock lk(data_mtx_);
  steps = hist_step_;
  chi2 = hist_chi2_;
}

std::vector<std::pair<std::string, double>> SimController::breakdown() const {
  std::scoped_lock lk(data_mtx_);
  return breakdown_;
}

std::string SimController::last_error() const {
  std::scoped_lock lk(data_mtx_);
  return last_error_;
}

void SimController::publish(RMC::Engine &eng, std::uint64_t step,
                            std::uint64_t acc, std::uint64_t tried, double chi2,
                            const RMC::AtomicStructure &s) {
  std::scoped_lock lk(data_mtx_);
  RenderSnapshot &w = latest_;
  const std::size_t n = s.size();
  w.natoms = n;
  w.xyz.resize(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto idx = static_cast<Eigen::Index>(i);
    w.xyz[3 * i + 0] = static_cast<float>(s.coordinates(idx, 0));
    w.xyz[3 * i + 1] = static_cast<float>(s.coordinates(idx, 1));
    w.xyz[3 * i + 2] = static_cast<float>(s.coordinates(idx, 2));
  }
  w.z.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    w.z[i] = s.atomic_numbers[static_cast<Eigen::Index>(i)];
  }
  w.box = box_flat_;
  w.step = step;
  w.accepted = acc;
  w.tried = tried;
  w.chi2 = chi2;
  w.generation = ++gen_counter_;

  hist_step_.push_back(static_cast<float>(step));
  hist_chi2_.push_back(static_cast<float>(chi2));

  breakdown_.clear();
  for (const auto &[name, val] : eng.constraints().error_breakdown()) {
    breakdown_.emplace_back(std::string(name), val);
  }
}

void SimController::worker_main(RMC::RMCConfig cfg) {
  std::optional<RMC::Engine> engine;
  std::string err;
  leaf::try_handle_all(
      [&]() -> leaf::result<void> {
        auto r = RMC::build_engine(cfg);
        if (!r) {
          return r.error();
        }
        engine.emplace(std::move(r.value()));
        return {};
      },
      [&](const std::string &msg) { err = msg; },
      [&](const leaf::error_info &) { err = "failed to build engine"; });

  if (!engine) {
    std::scoped_lock lk(data_mtx_);
    last_error_ = err.empty() ? "failed to build engine" : err;
    state_.store(RunState::Error);
    return;
  }

  RMC::Engine &eng = *engine;

  // The cell is constant during a run (RMC moves atoms, not the box) — capture
  // it once, column-major, as floats for the renderer.
  {
    std::scoped_lock lk(data_mtx_);
    const RMC::mat3_t b = RMC::periodic_box_or_zero(eng.boundary());
    for (int c = 0; c < 3; ++c) {
      for (int rr = 0; rr < 3; ++rr) {
        box_flat_[static_cast<std::size_t>(c * 3 + rr)] =
            static_cast<float>(b(rr, c));
      }
    }
  }

  eng.set_step_callback(
      [this, &eng](std::uint64_t step, std::uint64_t acc, std::uint64_t tried,
                   double chi2, const RMC::AtomicStructure &s) {
        publish(eng, step, acc, tried, chi2, s);
      },
      cfg.log_every == 0 ? 1 : cfg.log_every);

  state_.store(RunState::Running);

  constexpr std::uint64_t kChunk = 256; // cancel/pause granularity
  const std::uint64_t goal = cfg.steps; // 0 ⇒ run until stopped
  std::uint64_t done = 0;
  while (!stop_req_.load()) {
    if (pause_req_.load()) {
      state_.store(RunState::Paused);
      wait_while_paused();
      if (stop_req_.load()) {
        break;
      }
      state_.store(RunState::Running);
    }
    if (goal != 0 && done >= goal) {
      break;
    }
    std::uint64_t n = kChunk;
    if (goal != 0) {
      n = std::min<std::uint64_t>(kChunk, goal - done);
    }
    eng.run(n);
    done += n;
  }

  // Force a final snapshot so the last state is always shown.
  publish(eng, eng.steps_total(), eng.steps_accepted(), eng.stats().steps_tried,
          eng.total_error(), eng.structure());
  state_.store(RunState::Finished);
}

} // namespace rmcgui
