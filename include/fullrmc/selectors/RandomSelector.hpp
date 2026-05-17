#pragma once
#include <fullrmc/selectors/GroupSelector.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

namespace fullrmc {

struct RandomSelector : IGroupSelector {
    mutable boost::random::mt19937 rng;

    explicit RandomSelector(std::uint32_t seed = 42) : rng(seed) {}

    std::size_t select(std::size_t n_groups) override {
        boost::random::uniform_int_distribution<std::size_t> dist(0, n_groups - 1);
        return dist(rng);
    }

    std::unique_ptr<IGroupSelector> clone() const override {
        return std::make_unique<RandomSelector>(*this);
    }
};

// Selects groups with per-group weights via boost::random::discrete_distribution.
struct WeightedRandomSelector : IGroupSelector {
    mutable boost::random::mt19937 rng;
    std::vector<real_t>            weights;

    WeightedRandomSelector() = default;
    explicit WeightedRandomSelector(std::vector<real_t> w, std::uint32_t seed = 42)
        : rng(seed), weights(std::move(w)) {}

    void set_weights(std::span<const real_t> w) {
        weights.assign(w.begin(), w.end());
    }

    std::size_t select(std::size_t n_groups) override {
        if (weights.size() != n_groups) {
            // Fall back to uniform if weights not initialised yet.
            boost::random::uniform_int_distribution<std::size_t> u(0, n_groups - 1);
            return u(rng);
        }
        boost::random::discrete_distribution<std::size_t> dist(
            weights.begin(), weights.end());
        return dist(rng);
    }

    std::unique_ptr<IGroupSelector> clone() const override {
        return std::make_unique<WeightedRandomSelector>(*this);
    }
};

} // namespace fullrmc
