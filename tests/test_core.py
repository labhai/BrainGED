from __future__ import annotations

import numpy as np
import pytest

from brainged import AtlasMetadata, NodeMetadata, brain_ged
from brainged.core import (
    EPSILON,
    edge_discrepancy,
    hierarchy_distance,
    node_edit_cost,
    sparsify_top_k,
)


@pytest.fixture
def atlas() -> AtlasMetadata:
    return AtlasMetadata(
        [
            NodeMetadata("a", "L", ("visual-a", "visual")),
            NodeMetadata("b", "L", ("visual-a", "visual")),
            NodeMetadata("c", "L", ("visual-b", "visual")),
            NodeMetadata("d", "R", ("visual-a", "visual")),
            NodeMetadata("e", "L", ("default-a", "default")),
        ]
    )


def fc(size: int, edges: dict[tuple[int, int], float]) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=float)
    for (first, second), weight in edges.items():
        matrix[first, second] = matrix[second, first] = weight
    return matrix


def test_hierarchy_distance_has_the_paper_values(atlas: AtlasMetadata) -> None:
    assert hierarchy_distance(atlas[0], atlas[0]) == 0.0
    assert hierarchy_distance(atlas[0], atlas[1]) == pytest.approx(1 / 3)
    assert hierarchy_distance(atlas[0], atlas[2]) == pytest.approx(2 / 3)
    assert hierarchy_distance(atlas[0], atlas[4]) == 1.0


def test_arbitrary_depth_distance_and_substitution_gate() -> None:
    deep_atlas = AtlasMetadata(
        [
            NodeMetadata("a", "L", ("fine-a", "middle", "coarse")),
            NodeMetadata("b", "L", ("fine-b", "middle", "coarse")),
            NodeMetadata("c", "R", ("fine-c", "middle", "coarse")),
            NodeMetadata("d", "L", ("fine-d", "other-middle", "other-coarse")),
        ]
    )
    assert hierarchy_distance(deep_atlas[0], deep_atlas[1]) == pytest.approx(2 / 4)
    assert node_edit_cost([0], [1], deep_atlas) == pytest.approx(1 / 2)
    assert node_edit_cost([0], [2], deep_atlas) == 1.0
    assert node_edit_cost([0], [3], deep_atlas) == 1.0


def test_node_cost_uses_hierarchy_and_hemisphere_gate(atlas: AtlasMetadata) -> None:
    assert node_edit_cost([0], [0], atlas) == 0.0
    assert node_edit_cost([0], [1], atlas) == pytest.approx(1 / 3)
    assert node_edit_cost([0], [2], atlas) == pytest.approx(2 / 3)
    assert node_edit_cost([0], [3], atlas) == 1.0
    assert node_edit_cost([0], [4], atlas) == 1.0
    assert node_edit_cost([], [0], atlas) == 1.0
    assert node_edit_cost([], [], atlas) == 0.0


def test_node_cost_handles_unequal_retained_supports(atlas: AtlasMetadata) -> None:
    assert node_edit_cost([0, 1, 2], [0, 1], atlas) == pytest.approx(1 / 5)


def test_top_k_is_global_undirected_signed_and_deterministic() -> None:
    matrix = fc(4, {(0, 1): 0.2, (0, 2): -0.9, (1, 2): 0.8, (2, 3): 0.1})
    sparse = sparsify_top_k(matrix, 2)
    expected = fc(4, {(0, 2): -0.9, (1, 2): 0.8})
    np.testing.assert_array_equal(sparse.matrix, expected)
    assert sparse.nodes == (0, 1, 2)

    tied = sparsify_top_k(fc(3, {(0, 1): 0.9, (0, 2): -0.9}), 1)
    np.testing.assert_array_equal(tied.matrix, fc(3, {(0, 1): 0.9}))


def test_edge_discrepancy_uses_fixed_full_atlas_order() -> None:
    left = fc(3, {(0, 1): 0.8})
    right = fc(3, {(0, 2): 0.8})
    assert edge_discrepancy(left, right, denominator_epsilon=0.0) == 1.0
    assert edge_discrepancy(
        left, fc(3, {(0, 1): 0.4}), denominator_epsilon=0.0
    ) == pytest.approx(1 / 3)
    assert (
        edge_discrepancy(
            left, fc(3, {(0, 1): -0.8}), denominator_epsilon=0.0
        )
        == 1.0
    )


def test_edge_discrepancy_uses_the_paper_epsilon_by_default() -> None:
    left = fc(2, {(0, 1): 0.8})
    right = fc(2, {(0, 1): 0.4})
    assert edge_discrepancy(left, right) == pytest.approx(
        0.4 / (1.2 + EPSILON), rel=0.0, abs=1e-15
    )


def test_final_score_matches_the_two_component_formula(atlas: AtlasMetadata) -> None:
    left = fc(5, {(0, 1): 0.8})
    right = fc(5, {(0, 2): 0.8})
    result = brain_ged(left, right, atlas, top_k=1)
    assert result.node_cost == pytest.approx(1 / 3)
    assert result.edge_discrepancy == pytest.approx(1.0)
    assert result.similarity == pytest.approx(
        1.0 - 0.5 * (result.node_cost + result.edge_discrepancy)
    )
    assert result.source_nodes == ("a", "b")
    assert result.target_nodes == ("a", "c")


def test_identity_and_bounds(atlas: AtlasMetadata) -> None:
    left = fc(5, {(0, 1): 0.8, (1, 2): -0.6, (3, 4): 0.3})
    right = fc(5, {(0, 2): -0.7, (1, 2): 0.5, (3, 4): 0.4})
    identity = brain_ged(left, left, atlas, top_k=2)
    result = brain_ged(left, right, atlas, top_k=2)
    assert identity.similarity == 1.0
    assert identity.node_cost == identity.edge_discrepancy == 0.0
    assert 0.0 <= result.similarity <= 1.0


def test_all_isolates_have_unit_similarity(atlas: AtlasMetadata) -> None:
    result = brain_ged(np.zeros((5, 5)), np.zeros((5, 5)), atlas, top_k=0)
    assert result.similarity == 1.0
    assert result.node_cost == result.edge_discrepancy == 0.0
    assert result.source_nodes == result.target_nodes == ()


def test_atlas_rejects_mixed_hierarchy_depths() -> None:
    with pytest.raises(ValueError, match="same hierarchy depth"):
        AtlasMetadata(
            [
                NodeMetadata("a", "L", ("fine", "coarse")),
                NodeMetadata("b", "L", ("fine", "middle", "coarse")),
            ]
        )


@pytest.mark.parametrize(
    "left_hierarchy, right_hierarchy",
    [
        (
            ("shared-fine", "middle-a", "coarse"),
            ("shared-fine", "middle-b", "coarse"),
        ),
        (
            ("fine-a", "shared-middle", "coarse-a"),
            ("fine-b", "shared-middle", "coarse-b"),
        ),
    ],
)
def test_atlas_rejects_an_inconsistent_nested_hierarchy(
    left_hierarchy: tuple[str, ...], right_hierarchy: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="one parent"):
        AtlasMetadata(
            [
                NodeMetadata("a", "L", left_hierarchy),
                NodeMetadata("b", "L", right_hierarchy),
            ]
        )


@pytest.mark.parametrize(
    "operation, error",
    [
        (
            lambda atlas: brain_ged(
                np.zeros((2, 3)), np.zeros((2, 2)), atlas, top_k=1
            ),
            ValueError,
        ),
        (
            lambda atlas: brain_ged(
                np.array([[0.0, 0.5], [0.0, 0.0]]),
                np.zeros((2, 2)),
                atlas,
                top_k=1,
            ),
            ValueError,
        ),
        (lambda atlas: sparsify_top_k(np.zeros((2, 2)), 2), ValueError),
        (lambda atlas: node_edit_cost([0, 0], [], atlas), ValueError),
        (lambda atlas: sparsify_top_k([[0.0, np.nan], [np.nan, 0.0]], 1), ValueError),
        (lambda atlas: sparsify_top_k([[0.0, 1.0j], [-1.0j, 0.0]], 1), TypeError),
    ],
)
def test_invalid_inputs_fail(operation, error, atlas: AtlasMetadata) -> None:
    with pytest.raises(error):
        operation(atlas)
