#include <RMC/core/RandomStructure.hpp>
#include <RMC/core/RngGenerator.hpp>
#include <RMC/io/AtomicNumbers.hpp>

#include <boost/leaf/result.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace RMC {

Result<RandomStructure>
make_random_amorphous(std::span<const std::string> elements,
                      std::span<const std::size_t> counts, double spacing,
                      std::uint32_t seed) {
  if (elements.empty() || counts.size() != elements.size()) {
    return boost::leaf::new_error(
        std::string{"make_random_amorphous: elements and counts must match and "
                    "be non-empty"});
  } else if (!(spacing > 0.0)) {
    return boost::leaf::new_error(
        std::string{"make_random_amorphous: spacing must be positive"});
  }

  const std::size_t total =
      std::accumulate(counts.begin(), counts.end(), std::size_t{0});
  if (total == 0) {
    return boost::leaf::new_error(
        std::string{"make_random_amorphous: total atom count is zero"});
  }
  // Flat list of species indices, shuffled: random per-site species, exact counts.
  std::vector<std::size_t> species;
  species.reserve(total);
  for (auto [e, count] : counts | std::views::enumerate) {
    species.insert(species.end(), count, e);
  }
  RngBuffer<> rng(seed);
  auto &eng = rng.engine();
  for (std::size_t i = species.size(); i-- > 1;) {
    const std::size_t j = static_cast<std::size_t>(eng()) % (i + 1);
    std::swap(species[i], species[j]);
  }

  // Body-centred grid: two sites per cell (corner + centre).
  const int n =
      std::max(1, static_cast<int>(std::ceil(std::cbrt(
                      static_cast<double>(total) / 2.0))));
  const double cell = static_cast<double>(n) * spacing;

  std::vector<vec3_t> frac;
  frac.reserve(total);
  const double inv_n = 1.0 / static_cast<double>(n);
  for (int i = 0; i < n && frac.size() < total; ++i)
    for (int j = 0; j < n && frac.size() < total; ++j)
      for (int k = 0; k < n && frac.size() < total; ++k) {

        frac.emplace_back(i * inv_n, j * inv_n, k * inv_n);
        if (frac.size() >= total) {
          break;
        }
        frac.emplace_back((i + 0.5) * inv_n, (j + 0.5) * inv_n,
                          (k + 0.5) * inv_n);
      }

  RandomStructure out;
  out.box = mat3_t::Zero();
  out.box(0, 0) = cell;
  out.box(1, 1) = cell;
  out.box(2, 2) = cell;

  AtomicStructure &s = out.structure;
  s.coordinates.resize(static_cast<Eigen::Index>(total), 3);
  s.elements.resize(total);
  s.names.resize(total);
  s.residues.resize(total);
  s.molecule_ids.assign(total, 1);
  std::vector<int> anum(total);

  for (std::size_t a = 0; a < total; ++a) {
    const std::size_t e = species[a];
    const std::string &sym = elements[e];
    s.coordinates.row(static_cast<Eigen::Index>(a)) =
        (frac[a] * cell).transpose();
    s.elements[a] = sym;
    s.names[a] = sym;
    s.residues[a] = sym;
    anum[a] = io::atomic_number(sym);
  }
  s.atomic_numbers = Eigen::Map<const ivec_t>(
      anum.data(), static_cast<Eigen::Index>(anum.size()));

  return out;
}

} // namespace RMC
