"""Samplers and selectors, mirroring tests/test_sampler.cpp and
tests/test_selectors.cpp."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

import rmc


def test_greedy_accepts_downhill_and_equal_rejects_uphill() -> None:
    greedy = rmc.GreedySampler()
    assert greedy.accept(2.0, 1.0)
    assert greedy.accept(2.0, 2.0)
    assert not greedy.accept(1.0, 2.0)


def test_greedy_tolerance_admits_small_uphill_moves() -> None:
    greedy = rmc.GreedySampler(tolerance=0.5)
    assert greedy.tolerance == 0.5
    assert greedy.accept(1.0, 1.4)
    assert not greedy.accept(1.0, 1.6)


def test_metropolis_threshold() -> None:
    hot = rmc.MetropolisSampler(1.0)
    assert hot.temperature == 1.0
    assert hot.accept(2.0, 1.0, u01=0.999)
    threshold = math.exp(-1.0)
    assert hot.accept(1.0, 2.0, u01=threshold - 1e-9)
    assert not hot.accept(1.0, 2.0, u01=threshold + 1e-9)
    assert not rmc.MetropolisSampler(0.0).accept(1.0, 1.1, u01=0.0)


def test_annealing_schedule_cools_geometrically() -> None:
    schedule = rmc.AnnealingSampler.Schedule(t0=2.0, cooling=0.5, interval=10, t_min=0.1)
    annealing = rmc.AnnealingSampler(schedule)
    assert annealing.temperature(0) == pytest.approx(2.0)
    assert annealing.temperature(10) == pytest.approx(1.0)
    assert annealing.temperature(25) == pytest.approx(0.5)
    assert annealing.temperature(10_000) == pytest.approx(0.1)
    assert annealing.accept(1.0, 1.5, step=0, u01=0.7)
    assert not annealing.accept(1.0, 1.5, step=1000, u01=0.7)


def test_schedule_is_a_described_struct() -> None:
    default = rmc.AnnealingSampler.Schedule()
    assert (default.t0, default.cooling, default.interval, default.t_min) == (1.0, 0.9, 10000, 1e-6)
    assert rmc.AnnealingSampler.Schedule(t0=3.0) == rmc.AnnealingSampler.Schedule(t0=3.0)


def test_random_selector_returns_valid_indices() -> None:
    selector = rmc.RandomSelector(seed=1)
    assert all(0 <= selector.select(7) < 7 for _ in range(200))
    with pytest.raises(ValueError):
        selector.select(0)


def test_ordered_selector_cycles() -> None:
    selector = rmc.OrderedSelector()
    assert [selector.select(3) for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]


def test_weighted_selector_prefers_heavy_groups() -> None:
    selector = rmc.WeightedRandomSelector([1.0, 9.0], seed=2)
    counts = Counter(selector.select(2) for _ in range(2000))
    assert counts[1] > 3 * counts[0]
    selector.weights = [9.0, 1.0]
    assert selector.weights == [9.0, 1.0]


def test_smart_selector_feedback_moves_weights() -> None:
    selector = rmc.SmartRandomSelector(1.5, seed=3)
    selector.select(4)
    selector.feedback(0, True)
    selector.feedback(1, False)
    weights = selector.weights
    assert weights[0] > weights[2] > weights[1]
    assert weights.sum() == pytest.approx(1.0)


def test_directional_order() -> None:
    centroids = np.array([[5.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    near = rmc.DirectionalOrderSelector([0, 0, 0], centroids)
    assert [near.select(3) for _ in range(3)] == [1, 2, 0]
    far = rmc.DirectionalOrderSelector([0, 0, 0], centroids, nearest_first=False)
    assert [far.select(3) for _ in range(3)] == [0, 2, 1]


@pytest.mark.parametrize(
    ("mode", "trigger"),
    [(rmc.RecursiveMode.Refine, True), (rmc.RecursiveMode.Explore, False)],
)
def test_recursive_selector_retries_after_its_trigger(mode: rmc.RecursiveMode, trigger: bool) -> None:
    selector = rmc.RecursiveGroupSelector(rmc.OrderedSelector(), mode, max_retries=2)
    first = selector.select(5)
    selector.feedback(first, trigger)
    assert [selector.select(5), selector.select(5)] == [first, first]
    assert selector.select(5) != first  # retries exhausted


def test_recursive_selector_nests_and_copies() -> None:
    inner = rmc.RecursiveGroupSelector(rmc.OrderedSelector(), max_retries=1)
    outer = rmc.RecursiveGroupSelector(inner, rmc.RecursiveMode.Explore)
    assert outer.mode == rmc.RecursiveMode.Explore
    assert 0 <= outer.select(3) < 3
    assert inner.select(3) == 0  # the outer holds its own copy


def test_recursive_copies_reproduce_the_inner_stream() -> None:
    """Each RecursiveGroupSelector deep-copies its inner selector, RNG included."""
    inner = rmc.RandomSelector(seed=5)
    a = rmc.RecursiveGroupSelector(inner, max_retries=0)
    b = rmc.RecursiveGroupSelector(inner, max_retries=0)
    assert [a.select(50) for _ in range(20)] == [b.select(50) for _ in range(20)]
