#include <RMC/callbacks/Chi2CollectorCallback.hpp>

#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <ranges>
#include <string>
#include <string_view>

namespace RMC::callbacks {

void Chi2CollectorCallback::finalize() {
  if (finalized_ || history_.empty()) {
    finalized_ = true;
    return;
  }
  finalized_ = true;
  write_csv();
  print_ascii();
}

void Chi2CollectorCallback::write_csv() const {
  std::ofstream f(csv_path_);
  if (!f) {
    return;
  }
  f << "step,chi2\n";
  for (auto [step, chi2] : history_) {
    f << std::format("{},{:.10g}\n", step, chi2);
  }
}

void Chi2CollectorCallback::print_ascii() const {
  constexpr int bar_width = 40;
  const double chi2_max =
      std::ranges::max(history_, {}, &std::pair<std::uint64_t, double>::second)
          .second;

  std::cout << std::format("\nChi² convergence ({} samples):\n",
                           history_.size());
  for (auto [step, chi2] : history_) {
    const int bars =
        (chi2_max > 0.0) ? static_cast<int>(bar_width * chi2 / chi2_max) : 0;
    const auto blocks = std::views::repeat(std::string_view{"\xe2\x96\x88"}, bars) |
                        std::views::join | std::ranges::to<std::string>(); // U+2588 FULL BLOCK
    std::cout << std::format("step {:>10}  {:.4f}  {}\n", step, chi2, blocks);
  }
  std::cout << '\n';
}

} // namespace RMC::callbacks
