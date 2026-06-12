// angular_distribution — fit the bond-angle distribution (ADF), the angular
// companion to the PDF and the second descriptor of the Special Glass Structure
// method (Zhu et al., Acta Materialia 262 (2024) 119456).
//
// 1. Generate a random Cu-Zr cell (the "reference") and compute its ADF.
// 2. Generate a *different* random cell of the same composition.
// 3. Refine cell 2 with an AngularDistributionConstraint targeting the
//    reference ADF, and watch the total chi² fall.
#include <RMC/Engine.hpp>
#include <RMC/analysis/AngularDistribution.hpp>
#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/core/RandomStructure.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace RMC;

int main() {
  const std::vector<std::string> elements{"Cu", "Zr"};
  const std::vector<std::size_t> counts{64, 64};
  const double spacing = 2.8;

  // 1. Reference cell + its ADF (the fit target).
  auto ref = make_random_amorphous(elements, counts, spacing, /*seed=*/1);
  if (!ref) {
    std::cerr << "failed to generate reference structure\n";
    return 1;
  }
  analysis::AdfParams ap;
  ap.max_dis = 3.6;
  ap.n_bins = 60;
  auto target = analysis::compute_adf(ref->structure.coordinates,
                                      ref->periodic_bc(),
                                      ref->structure.elements, ap);
  if (!target) {
    std::cerr << "failed to compute reference ADF\n";
    return 1;
  }

  // Pack theta + the per-triplet partials into the constraint's data layout
  // (column 0 = angle, columns 1.. = triplet targets in canonical order).
  const int n_bins = static_cast<int>(target->theta.size());
  const int n_part = static_cast<int>(target->partials.size());
  mat_t adf(n_bins, 1 + n_part);
  adf.col(0) = target->theta;
  for (int j = 0; j < n_part; ++j)
    adf.col(1 + j) = target->partials[static_cast<std::size_t>(j)].values;
  std::cout << "ADF target: " << n_part << " triplets, " << n_bins << " bins\n";

  // 2. A different random cell of the same composition.
  auto cell = make_random_amorphous(elements, counts, spacing, /*seed=*/2);
  if (!cell) {
    std::cerr << "failed to generate working structure\n";
    return 1;
  }

  // 3. Refine its ADF toward the reference.
  Engine eng(cell->structure, cell->periodic_bc());

  AngularDistributionConstraint adc;
  adc.set_experimental_data(adf);
  adc.set_cutoff(ap.max_dis);
  adc.set_smoothing(ap.smooth_range);
  adc.set_elements(eng.structure().elements);
  adc.initialise();
  eng.add_constraint(std::move(adc));

  eng.build_atomic_groups(0.0, 0.15, /*seed=*/42);

  eng.set_step_callback(
      [](std::uint64_t step, std::uint64_t acc, std::uint64_t tried, double chi2,
         const AtomicStructure &) {
        const double rate =
            tried ? 100.0 * static_cast<double>(acc) / static_cast<double>(tried)
                  : 0.0;
        std::cout << "step " << step << "  acc=" << rate << "%  chi2=" << chi2
                  << "\n";
      },
      2000);

  eng.run(10000);

  std::cout << "done: accepted " << eng.stats().steps_accepted << " / "
            << eng.stats().steps_tried
            << "  final chi2=" << eng.stats().last_total_err << "\n";
  return 0;
}
