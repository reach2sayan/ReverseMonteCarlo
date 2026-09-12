#pragma once
#include <RMC/core/Structure.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
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
class Chi2CollectorCallback {
public:
  explicit Chi2CollectorCallback(std::filesystem::path csv_path = "chi2.csv")
      : csv_path_{std::move(csv_path)} {}

  void operator()(std::uint64_t step, std::uint64_t /*acc*/,
                  std::uint64_t /*tried*/, double chi2,
                  const AtomicStructure & /*s*/) {
    history_.emplace_back(step, chi2);
  }

  void finalize();

  ~Chi2CollectorCallback() { finalize(); }
  [[nodiscard]] const std::vector<std::pair<std::uint64_t, double>> &
  history() const noexcept {
    return history_;
  }

private:
  void write_csv() const;
  void print_ascii() const;

  std::vector<std::pair<std::uint64_t, double>> history_;
  std::filesystem::path csv_path_;
  bool finalized_{false};
};

} // namespace RMC::callbacks
