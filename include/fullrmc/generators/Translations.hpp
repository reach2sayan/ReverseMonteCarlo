#pragma once
#include <fullrmc/generators/MoveGenerator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <cmath>
#include <numbers>

namespace fullrmc {

// Translates all atoms in a group by the same random displacement vector.
// Amplitude is drawn uniformly from [min_amp, max_amp] (Angstrom).
// Direction is uniform on the unit sphere via the Marsaglia method.
struct TranslationGenerator : MoveGeneratorBase<TranslationGenerator> {
    real_t min_amp{0.0};
    real_t max_amp{0.2};
    mutable boost::random::mt19937 rng;

    TranslationGenerator() = default;
    TranslationGenerator(real_t mn, real_t mx, std::uint32_t seed = 42)
        : min_amp(mn), max_amp(mx), rng(seed) {}

    void generate_impl(coords_t& coords,
                       std::span<const index_t> indices) {
        boost::random::uniform_real_distribution<real_t> amp_dist(min_amp, max_amp);
        vec3_t delta = random_unit_vector() * amp_dist(rng);
        for (auto i : indices)
            coords.row(i) += delta.transpose();
    }

private:
    vec3_t random_unit_vector() {
        // Marsaglia (1972) uniform sphere sampling
        boost::random::normal_distribution<real_t> nd(0.0, 1.0);
        vec3_t v(nd(rng), nd(rng), nd(rng));
        real_t n = v.norm();
        if (n < 1e-12) return vec3_t::UnitX();
        return v / n;
    }
};

// Translates along a fixed axis only.
struct TranslationAlongAxisGenerator : MoveGeneratorBase<TranslationAlongAxisGenerator> {
    vec3_t axis{1.0, 0.0, 0.0};
    real_t min_amp{0.0};
    real_t max_amp{0.2};
    mutable boost::random::mt19937 rng;

    TranslationAlongAxisGenerator() = default;
    TranslationAlongAxisGenerator(vec3_t ax, real_t mn, real_t mx,
                                   std::uint32_t seed = 42)
        : axis(ax.normalized()), min_amp(mn), max_amp(mx), rng(seed) {}

    void generate_impl(coords_t& coords,
                       std::span<const index_t> indices) {
        boost::random::uniform_real_distribution<real_t> amp_dist(min_amp, max_amp);
        boost::random::uniform_real_distribution<real_t> sign_dist(-1.0, 1.0);
        real_t magnitude = amp_dist(rng);
        if (sign_dist(rng) < 0.0) magnitude = -magnitude;
        vec3_t delta = axis * magnitude;
        for (auto i : indices)
            coords.row(i) += delta.transpose();
    }
};

// Translates towards a fixed centre point.
struct TranslationTowardsCentreGenerator : MoveGeneratorBase<TranslationTowardsCentreGenerator> {
    vec3_t centre{0.0, 0.0, 0.0};
    real_t min_amp{0.0};
    real_t max_amp{0.2};
    mutable boost::random::mt19937 rng;

    TranslationTowardsCentreGenerator() = default;
    TranslationTowardsCentreGenerator(vec3_t c, real_t mn, real_t mx,
                                       std::uint32_t seed = 42)
        : centre(c), min_amp(mn), max_amp(mx), rng(seed) {}

    void generate_impl(coords_t& coords,
                       std::span<const index_t> indices) {
        boost::random::uniform_real_distribution<real_t> amp_dist(min_amp, max_amp);
        vec3_t gc      = centroid(coords, indices);
        vec3_t dir     = (centre - gc);
        real_t dist    = dir.norm();
        if (dist < 1e-12) return;
        vec3_t delta = (dir / dist) * amp_dist(rng);
        for (auto i : indices)
            coords.row(i) += delta.transpose();
    }
};

} // namespace fullrmc
