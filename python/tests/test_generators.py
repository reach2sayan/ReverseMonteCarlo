"""Move generators, mirroring tests/test_generators.cpp, plus the holder,
Python-defined generators and Group."""

from __future__ import annotations

import gc
from itertools import combinations

import numpy as np
import pytest

import rmc


def rows(*atoms: tuple[float, float, float]) -> np.ndarray:
    return np.array(atoms, dtype=float)


def pairwise(coords: np.ndarray) -> list[float]:
    return [float(np.linalg.norm(a - b)) for a, b in combinations(coords, 2)]


def test_translation_amplitude_within_bounds() -> None:
    gen = rmc.TranslationGenerator(0.1, 0.5, seed=99)
    coords = rows((0, 0, 0), (1, 0, 0), (0, 1, 0))
    before = coords.copy()
    gen.generate(coords, [0, 1, 2])
    shift = coords - before
    assert 0.1 - 1e-9 <= np.linalg.norm(shift[0]) <= 0.5 + 1e-9
    np.testing.assert_allclose(shift, np.broadcast_to(shift[0], shift.shape), atol=1e-12)


def test_translation_of_a_subset() -> None:
    gen = rmc.TranslationGenerator(0.0, 0.2, seed=7)
    coords = rows((0, 0, 0), (1, 0, 0), (2, 0, 0))
    gen.generate(coords, [0, 2])
    np.testing.assert_allclose(coords[1], [1, 0, 0])


@pytest.mark.parametrize(
    "gen",
    [
        rmc.RotationGenerator(0.1, 0.5, seed=3),
        rmc.RotationAboutSymmetryAxisGenerator(rmc.SymmetryAxis.Z, 0.1, 0.5),
        rmc.OrientationGenerator([0.0, 0.0, 1.0], 0.2),
    ],
    ids=["random-axis", "symmetry-axis", "orientation"],
)
def test_rotations_preserve_pairwise_distances(gen: object) -> None:
    coords = rows((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    before = pairwise(coords)
    gen.generate(coords, [0, 1, 2, 3])  # type: ignore[attr-defined]
    np.testing.assert_allclose(pairwise(coords), before, atol=1e-10)


def test_swap_exchanges_positions() -> None:
    gen = rmc.SwapGenerator([[2, 3]])
    coords = rows((0, 0, 0), (1, 0, 0), (5, 0, 0), (6, 0, 0))
    before = coords.copy()
    gen.generate(coords, [0, 1])
    np.testing.assert_allclose(coords[[0, 1]], before[[2, 3]])
    np.testing.assert_allclose(coords[[2, 3]], before[[0, 1]])


def test_translation_towards_axis_moves_centroid_closer() -> None:
    gen = rmc.TranslationTowardsAxisGenerator([0, 0, 0], [0, 0, 1], 0.1, 0.5, seed=1)
    coords = rows((3, 0, 0), (3, 1, 0))
    before = np.linalg.norm(coords.mean(axis=0)[:2])
    gen.generate(coords, [0, 1])
    assert np.linalg.norm(coords.mean(axis=0)[:2]) < before


def test_symmetry_axis_generators_are_axis_generators() -> None:
    gen = rmc.TranslationAlongSymmetryAxisGenerator(rmc.SymmetryAxis.Z, 0.1, 0.2)
    assert isinstance(gen, rmc.TranslationAlongAxisGenerator)
    coords = rows((0, 0, 0))
    gen.generate(coords, [0])
    np.testing.assert_allclose(coords[0, :2], [0, 0])
    assert 0.1 - 1e-9 <= abs(coords[0, 2]) <= 0.2 + 1e-9


def test_distance_agitation_keeps_the_midpoint() -> None:
    gen = rmc.DistanceAgitationGenerator(0, 1, 0.1, 0.2, seed=5)
    coords = rows((0, 0, 0), (2, 0, 0))
    gen.generate(coords, [0, 1])
    np.testing.assert_allclose(coords.mean(axis=0), [1, 0, 0], atol=1e-12)
    assert abs(np.linalg.norm(coords[1] - coords[0]) - 2.0) >= 0.1 - 1e-9


def test_angle_agitation_keeps_bond_lengths() -> None:
    gen = rmc.AngleAgitationGenerator(0, 1, 2, 0.1, 0.2, seed=5)
    coords = rows((1, 0, 0), (0, 0, 0), (0, 1, 0))
    gen.generate(coords, [0, 1, 2])
    np.testing.assert_allclose(
        [np.linalg.norm(coords[0] - coords[1]), np.linalg.norm(coords[2] - coords[1])],
        [1.0, 1.0],
        atol=1e-12,
    )


def test_swap_centers_moves_the_centroid() -> None:
    gen = rmc.SwapCentersGenerator([[2, 3]])
    coords = rows((0, 0, 0), (1, 0, 0), (5, 0, 0), (7, 0, 0))
    gen.generate(coords, [0, 1])
    np.testing.assert_allclose(coords[[0, 1]].mean(axis=0), [6, 0, 0])


def test_translation_path_cycles() -> None:
    gen = rmc.TranslationAlongAxisPath([1.0, 0.0, 0.0], [0.5, -0.5])
    coords = rows((0, 0, 0))
    gen.generate(coords, [0])
    np.testing.assert_allclose(coords[0], [0.5, 0, 0])
    gen.generate(coords, [0])
    np.testing.assert_allclose(coords[0], [0, 0, 0], atol=1e-12)


def test_rotation_path_full_cycle_is_identity() -> None:
    gen = rmc.RotationAboutAxisPath([0.0, 0.0, 1.0], [np.pi / 2] * 4)
    coords = rows((1, 0, 0), (-1, 0, 0), (0, 2, 0))
    before = coords.copy()
    for _ in range(4):
        gen.generate(coords, [0, 1, 2])
    np.testing.assert_allclose(coords, before, atol=1e-12)


def test_collector_applies_exactly_one_generator() -> None:
    collector = rmc.MoveGeneratorCollector(seed=1)
    collector.add(rmc.TranslationAlongSymmetryAxisGenerator(rmc.SymmetryAxis.X, 1.0, 1.0))
    collector.add(rmc.TranslationAlongSymmetryAxisGenerator(rmc.SymmetryAxis.Y, 1.0, 1.0), 2.0)
    for _ in range(10):
        coords = rows((0, 0, 0))
        collector.generate(coords, [0])
        assert sorted(np.abs(coords[0])) == pytest.approx([0.0, 0.0, 1.0])


def test_amplitude_is_a_described_field() -> None:
    gen = rmc.TranslationGenerator(0.1, 0.5)
    assert gen.amp == rmc.Amplitude(lo=0.1, hi=0.5)
    gen.amp = rmc.Amplitude(lo=0.3, hi=0.3)
    coords = rows((0, 0, 0))
    gen.generate(coords, [0])
    assert np.linalg.norm(coords[0]) == pytest.approx(0.3)
    assert rmc.RotationGenerator(0.0, 0.1).angle == rmc.Amplitude(lo=0.0, hi=0.1)


def test_seed_is_keyword_only_and_indices_are_checked() -> None:
    with pytest.raises(TypeError):
        rmc.TranslationGenerator(0.1, 0.5, 3)  # type: ignore[misc]
    with pytest.raises(IndexError):
        rmc.TranslationGenerator(0.1, 0.5).generate(rows((0, 0, 0)), [1])


def test_species_swap_swaps_on_the_borrowed_structure() -> None:
    gen = rmc.SpeciesSwapGenerator(
        rmc.AtomicStructure(np.zeros((4, 3)), ["Cu", "Au", "Cu", "Au"]), [[0, 1, 2, 3]], seed=3
    )
    gc.collect()  # the structure lives on through the generator
    assert rmc.MoveGenerator(gen).modifies_species


def test_species_swap_exchanges_species() -> None:
    atoms = rmc.AtomicStructure(np.zeros((4, 3)), ["Cu", "Au", "Cu", "Au"])
    gen = rmc.SpeciesSwapGenerator(atoms, [[0, 1, 2, 3]], seed=3)
    gen.generate(np.zeros((4, 3)), [0])
    assert atoms.elements[0] == "Au"
    assert sorted(atoms.elements) == ["Au", "Au", "Cu", "Cu"]
    assert atoms.atomic_numbers.tolist().count(79) == 2


# ---- the holder, Group, Python-defined generators ----


class Nudge:
    """Moves the given atoms +1 along x."""

    def generate(self, coords: np.ndarray, indices: np.ndarray) -> None:
        assert coords.flags.writeable
        coords[indices, 0] += 1.0


def test_group_converts_generators_implicitly() -> None:
    group = rmc.Group("all", [0, 1, 2], rmc.TranslationGenerator(0.1, 0.2))
    assert isinstance(group.generator, rmc.MoveGenerator)
    assert len(group) == 3
    assert group.refine
    assert repr(group) == "Group('all', 3 atoms)"


def test_frozen_group_without_generator() -> None:
    group = rmc.Group("frozen", [0], refine=False)
    assert group.generator is None
    assert repr(group) == "Group('frozen', 1 atoms, no generator, frozen)"
    with pytest.raises(TypeError):
        rmc.Group("x", [0], None, False)  # type: ignore[misc]


def test_arbitrary_objects_do_not_convert() -> None:
    with pytest.raises(TypeError):
        rmc.Group("g", [0], Nudge())  # type: ignore[arg-type]


def test_python_generator_writes_through() -> None:
    held = rmc.MoveGenerator(Nudge())
    coords = rows((0, 0, 0), (0, 0, 0), (0, 0, 0))
    held.generate(coords, [0, 2])
    np.testing.assert_allclose(coords[:, 0], [1, 0, 1])
    assert not held.modifies_species
    assert held.rejection_override() is None
    assert rmc.Group("g", [0], held).generator is not None


def test_python_generator_optional_hooks() -> None:
    class Judge(Nudge):
        modifies_species = True

        def rejection_override(self) -> bool | None:
            return True

    held = rmc.MoveGenerator(Judge())
    assert held.modifies_species
    assert held.rejection_override() is True


def test_python_generator_needs_generate() -> None:
    with pytest.raises(TypeError, match="generate"):
        rmc.MoveGenerator(object())


def test_gradient_generators_follow_the_constraints() -> None:
    class Well:
        def compute_error(self, coords: np.ndarray, moved: np.ndarray) -> float:
            return float(np.sum(coords[:, 0] ** 2))

    constraints = rmc.ConstraintCollection()
    constraints.add(rmc.Constraint(Well()))
    for gen in (
        rmc.LangevinTranslationGenerator(0.01, constraints, seed=1),
        rmc.LangevinRotationGenerator(0.01, constraints, seed=1),
    ):
        coords = rows((1, 0, 0), (2, 0, 0))
        gen.generate(coords, [0, 1])
        assert np.isfinite(coords).all()
        assert not np.array_equal(coords, rows((1, 0, 0), (2, 0, 0)))

    leapfrog = rmc.MoveGenerator(rmc.LeapfrogTranslationGenerator(5, 0.01, constraints, seed=2))
    leapfrog.generate(rows((1, 0, 0), (2, 0, 0)), [0, 1])
    assert leapfrog.rejection_override() in (True, False)
