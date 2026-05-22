#pragma once
#include <RMC/core/Structure.hpp>

#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace RMC::callbacks {

// Accumulates (step, chi²) pairs from the chi2 callback param and, at the
// end, writes a CSV file and prints an ASCII convergence chart to stdout.
//
// Because this callback is stateful it must be passed via std::ref so that
// the std::function wrapper does not copy the accumulated history:
//
//   Chi2CollectorCallback collector("chi2.csv");
//   engine.set_step_callback(std::ref(collector), 1000);
//   engine.run(N);
//   collector.finalize();   // explicit call, or let the destructor handle it
//
// finalize() is idempotent — safe to call multiple times.
class Chi2CollectorCallback {
public:
  explicit Chi2CollectorCallback(std::filesystem::path csv_path = "chi2.csv")
      : csv_path_(std::move(csv_path)) {}

  constexpr void operator()(std::uint64_t step, std::uint64_t /*acc*/,
                  std::uint64_t /*tried*/, double chi2,
                  const AtomicStructure & /*s*/) {
    history_.emplace_back(step, chi2);
  }

  // Write CSV + print ASCII chart. Safe to call more than once.
  void finalize() {
    if (finalized_ || history_.empty()) {
      finalized_ = true;
      return;
    }
    finalized_ = true;
    write_csv();
    print_ascii();
  }

  ~Chi2CollectorCallback() { finalize(); }

  [[nodiscard]] const std::vector<std::pair<std::uint64_t, double>> &
  history() const noexcept {
    return history_;
  }

private:
  void write_csv() const {
    std::ofstream f(csv_path_);
    if (!f) {
      return;
    }
    f << "step,chi2\n";
    for (auto [step, chi2] : history_) {
      f << std::format("{},{:.10g}\n", step, chi2);
    }
  }

  void print_ascii() const {
    constexpr int bar_width = 40;
    const double chi2_max =
        std::ranges::max(history_, {},
                         &std::pair<std::uint64_t, double>::second)
            .second;

    std::cout << std::format("\nChi² convergence ({} samples):\n",
                             history_.size());
    for (auto [step, chi2] : history_) {
      const int bars =
          (chi2_max > 0.0) ? static_cast<int>(bar_width * chi2 / chi2_max) : 0;
      std::string blocks;
      blocks.reserve(static_cast<std::size_t>(bars) * 3);
      for (int i = 0; i < bars; ++i) {
        blocks += "\xe2\x96\x88"; // U+2588 FULL BLOCK
      }
      std::cout << std::format("step {:>10}  {:.4f}  {}\n", step, chi2, blocks);
    }
    std::cout << '\n';
  }

  std::vector<std::pair<std::uint64_t, double>> history_;
  std::filesystem::path csv_path_;
  bool finalized_{false};
};

} // namespace RMC::callbacks
