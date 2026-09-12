#include <RMC/core/RandomStructure.hpp>
#include <RMC/core/RngGenerator.hpp>

#include <boost/leaf/result.hpp>
#include <seitz/data/element_data.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <ranges>
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
      std::ranges::fold_left(counts, std::size_t{0}, std::plus{});
  if (total == 0) {
    return boost::leaf::new_error(
        std::string{"make_random_amorphous: total atom count is zero"});
  }
  // Species index per site, shuffled: random per-site species, exact counts.
  auto species = counts | std::views::enumerate |
                 std::views::transform([](auto ec) {
                   const auto [e, count] = ec;
                   return std::views::repeat(static_cast<std::size_t>(e), count);
                 }) |
                 std::views::join | std::ranges::to<std::vector>();
  Rng rng{seed};
  std::ranges::shuffle(species, rng.engine());

  // Body-centred grid in lattice units: two sites per cell (corner + centre).
  const int n = std::max(
      1, static_cast<int>(std::ceil(std::cbrt(static_cast<double>(total) / 2.0))));
  const auto axis = std::views::iota(0, n);
  auto sites = std::views::cartesian_product(axis, axis, axis) |
               std::views::transform([](auto ijk) {
                 const auto [i, j, k] = ijk;
                 const vec3_t corner = Eigen::Vector3i(i, j, k).cast<double>();
                 return std::array{corner, vec3_t(corner.array() + 0.5)};
               }) |
               std::views::join | std::views::take(total);

  AtomicStructure s;
  s.coordinates.resize(static_cast<Eigen::Index>(total), 3);
  // The index type is the range's difference_type, which for this joined
  // cartesian product is wider than Eigen::Index on MSVC -- hence the cast.
  for (const auto [a, site] : sites | std::views::enumerate) {
    s.coordinates.row(static_cast<Eigen::Index>(a)) = spacing * site.transpose();
  }
  s.elements = species |
               std::views::transform([&](std::size_t e) { return elements[e]; }) |
               std::ranges::to<std::vector>();
  s.names = s.elements;
  s.residues = s.elements;
  s.molecule_ids.assign(total, 1);
  s.atomic_numbers = ivec_t::NullaryExpr(
      static_cast<Eigen::Index>(total), [&](Eigen::Index a) {
        return seitz::data::atomic_number(s.elements[static_cast<std::size_t>(a)])
            .value_or(0);
      });

  RandomStructure out{.structure = std::move(s), .box=mat3_t::Identity() * (n * spacing) };
  return out;
}

} // namespace RMC
