"""BrainGED for signed, undirected functional-connectivity graphs."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

EPSILON = 1e-12


def _label(value: Hashable, name: str) -> Hashable:
    if not isinstance(value, Hashable):
        raise TypeError(f"{name} must be hashable")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _hemisphere(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("hemisphere must be a string")
    aliases = {
        "L": "L",
        "LH": "L",
        "LEFT": "L",
        "R": "R",
        "RH": "R",
        "RIGHT": "R",
    }
    try:
        return aliases[value.strip().upper()]
    except KeyError as exc:
        raise ValueError("hemisphere must identify left or right") from exc


@dataclass(frozen=True, slots=True)
class NodeMetadata:
    """Parcel identity, hemisphere, and fine-to-coarse hierarchy labels."""

    roi_id: Hashable
    hemisphere: str
    hierarchy: tuple[Hashable, ...]

    def __init__(
        self,
        roi_id: Hashable,
        hemisphere: str,
        hierarchy: Iterable[Hashable],
    ) -> None:
        if isinstance(hierarchy, (str, bytes)):
            raise TypeError("hierarchy must be an iterable of labels")
        try:
            labels = tuple(hierarchy)
        except TypeError as exc:
            raise TypeError("hierarchy must be an iterable of labels") from exc
        if not labels:
            raise ValueError("hierarchy must contain at least one community label")

        object.__setattr__(self, "roi_id", _label(roi_id, "roi_id"))
        object.__setattr__(self, "hemisphere", _hemisphere(hemisphere))
        object.__setattr__(
            self,
            "hierarchy",
            tuple(
                _label(label, f"hierarchy[{index}]")
                for index, label in enumerate(labels)
            ),
        )

    @property
    def levels(self) -> tuple[Hashable, ...]:
        """All Eq. (1) labels, from parcel identity to coarsest community."""

        return (self.roi_id, *self.hierarchy)

    @property
    def coarsest(self) -> Hashable:
        return self.hierarchy[-1]


@dataclass(frozen=True, slots=True)
class AtlasMetadata:
    """Metadata in the same parcel order as the input FC matrices."""

    nodes: tuple[NodeMetadata, ...]

    def __init__(self, nodes: Iterable[NodeMetadata]):
        values = tuple(nodes)
        if not all(isinstance(node, NodeMetadata) for node in values):
            raise TypeError("atlas entries must be NodeMetadata instances")
        roi_ids = [node.roi_id for node in values]
        if len(set(roi_ids)) != len(roi_ids):
            raise ValueError("roi_id values must be unique")

        depths = {len(node.levels) for node in values}
        if len(depths) > 1:
            raise ValueError("all atlas entries must use the same hierarchy depth")

        if values:
            for level in range(len(values[0].levels) - 1):
                parent_by_child: dict[Hashable, Hashable] = {}
                for node in values:
                    child = node.levels[level]
                    parent = node.levels[level + 1]
                    known_parent = parent_by_child.setdefault(child, parent)
                    if known_parent != parent:
                        raise ValueError("hierarchy labels must have one parent per level")
        object.__setattr__(self, "nodes", values)

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, index: int) -> NodeMetadata:
        return self.nodes[index]


@dataclass(frozen=True, slots=True)
class SparsifiedFC:
    """Full-atlas top-k matrix and its retained parcel indices."""

    matrix: NDArray[np.float64]
    nodes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BrainGEDResult:
    """BrainGED score and its two normalized components."""

    similarity: float
    node_cost: float
    edge_discrepancy: float
    source_nodes: tuple[Hashable, ...]
    target_nodes: tuple[Hashable, ...]


def _validate_fc(value: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError("FC matrix must be real-valued")
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("FC matrix must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"FC matrix must be square, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("FC matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError("FC matrix must be symmetric")
    if np.max(np.abs(matrix), initial=0.0) > 1.0 + 1e-12:
        raise ValueError("FC weights must lie in [-1, 1]")

    matrix = np.array(0.5 * (matrix + matrix.T), dtype=np.float64, copy=True)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _epsilon(value: float, name: str, *, positive: bool) -> float:
    value = float(value)
    if not math.isfinite(value) or (value <= 0.0 if positive else value < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def sparsify_top_k(fc: ArrayLike, top_k: int) -> SparsifiedFC:
    """Retain the global absolute top-k unique edges and preserve their signs."""

    if isinstance(top_k, bool) or not isinstance(top_k, Integral):
        raise TypeError("top_k must be an integer")
    top_k = int(top_k)
    matrix = _validate_fc(fc)
    n_nodes = matrix.shape[0]
    maximum = n_nodes * (n_nodes - 1) // 2
    if top_k < 0 or top_k > maximum:
        raise ValueError(f"top_k must be between 0 and {maximum}")

    first, second = np.triu_indices(n_nodes, k=1)
    weights = matrix[first, second]
    order = np.lexsort((second, first, -np.abs(weights)))
    chosen = order[:top_k]
    selected_first = first[chosen]
    selected_second = second[chosen]

    sparse = np.zeros_like(matrix)
    sparse[selected_first, selected_second] = weights[chosen]
    sparse[selected_second, selected_first] = weights[chosen]
    retained = tuple(
        int(index)
        for index in np.unique(np.concatenate((selected_first, selected_second)))
    )
    sparse.setflags(write=False)
    return SparsifiedFC(sparse, retained)


def hierarchy_distance(left: NodeMetadata, right: NodeMetadata) -> float:
    """Return the Eq. (1) mismatch fraction across all hierarchy levels."""

    if not isinstance(left, NodeMetadata) or not isinstance(right, NodeMetadata):
        raise TypeError("left and right must be NodeMetadata instances")
    if len(left.levels) != len(right.levels):
        raise ValueError("nodes must use the same hierarchy depth")
    mismatches = sum(
        left_label != right_label
        for left_label, right_label in zip(left.levels, right.levels, strict=True)
    )
    return mismatches / len(left.levels)


def _substitution_cost(
    left: NodeMetadata, right: NodeMetadata, epsilon: float
) -> float:
    allowed = left.hemisphere == right.hemisphere and left.coarsest == right.coarsest
    return 2.0 * hierarchy_distance(left, right) if allowed else 2.0 + epsilon


def _indices(values: Sequence[int], atlas_size: int, name: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain integer parcel indices")
        index = int(value)
        if index < 0 or index >= atlas_size:
            raise IndexError(f"{name} contains out-of-range parcel index {index}")
        parsed.append(index)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicate parcel indices")
    return tuple(sorted(parsed))


def node_edit_cost(
    source_nodes: Sequence[int],
    target_nodes: Sequence[int],
    atlas: AtlasMetadata,
    *,
    forbidden_epsilon: float = EPSILON,
) -> float:
    """Compute normalized node edit cost with full dummy-node assignment."""

    if not isinstance(atlas, AtlasMetadata):
        raise TypeError("atlas must be an AtlasMetadata instance")
    forbidden_epsilon = _epsilon(
        forbidden_epsilon, "forbidden_epsilon", positive=True
    )
    source = _indices(source_nodes, len(atlas), "source_nodes")
    target = _indices(target_nodes, len(atlas), "target_nodes")
    n_source, n_target = len(source), len(target)
    if n_source == 0 and n_target == 0:
        return 0.0

    size = n_source + n_target
    costs = np.full((size, size), np.inf, dtype=np.float64)
    for row, source_index in enumerate(source):
        for column, target_index in enumerate(target):
            costs[row, column] = _substitution_cost(
                atlas[source_index], atlas[target_index], forbidden_epsilon
            )

    source_range = np.arange(n_source)
    target_range = np.arange(n_target)
    costs[source_range, n_target + source_range] = 1.0
    costs[n_source + target_range, target_range] = 1.0
    costs[n_source:, n_target:] = 0.0

    rows, columns = linear_sum_assignment(costs)
    normalized = float(costs[rows, columns].sum()) / float(size)
    return min(1.0, max(0.0, normalized))


def edge_discrepancy(
    source: ArrayLike,
    target: ArrayLike,
    *,
    denominator_epsilon: float = EPSILON,
) -> float:
    """Compute Eq. (5) once over the fixed-atlas upper triangle."""

    denominator_epsilon = _epsilon(
        denominator_epsilon, "denominator_epsilon", positive=False
    )
    left = _validate_fc(source)
    right = _validate_fc(target)
    if left.shape != right.shape:
        raise ValueError("FC matrices must have the same shape")
    first, second = np.triu_indices(left.shape[0], k=1)
    left_edges = left[first, second]
    right_edges = right[first, second]
    numerator = float(np.abs(left_edges - right_edges).sum())
    denominator = (
        float((np.abs(left_edges) + np.abs(right_edges)).sum())
        + denominator_epsilon
    )
    if denominator == 0.0:
        return 0.0
    return min(1.0, max(0.0, numerator / denominator))


def brain_ged(
    source_fc: ArrayLike,
    target_fc: ArrayLike,
    atlas: AtlasMetadata,
    *,
    top_k: int,
    forbidden_epsilon: float = EPSILON,
    denominator_epsilon: float = EPSILON,
) -> BrainGEDResult:
    """Compute BrainGED similarity according to Eqs. (1)-(6)."""

    if not isinstance(atlas, AtlasMetadata):
        raise TypeError("atlas must be an AtlasMetadata instance")
    forbidden_epsilon = _epsilon(
        forbidden_epsilon, "forbidden_epsilon", positive=True
    )
    denominator_epsilon = _epsilon(
        denominator_epsilon, "denominator_epsilon", positive=False
    )
    source = sparsify_top_k(source_fc, top_k)
    target = sparsify_top_k(target_fc, top_k)
    if source.matrix.shape != target.matrix.shape:
        raise ValueError("FC matrices must have the same shape")
    if source.matrix.shape[0] != len(atlas):
        raise ValueError("atlas length must match the FC matrix dimensions")

    node_cost = node_edit_cost(
        source.nodes,
        target.nodes,
        atlas,
        forbidden_epsilon=forbidden_epsilon,
    )
    discrepancy = edge_discrepancy(
        source.matrix,
        target.matrix,
        denominator_epsilon=denominator_epsilon,
    )
    similarity = 1.0 - 0.5 * (node_cost + discrepancy)
    similarity = min(1.0, max(0.0, similarity))
    return BrainGEDResult(
        similarity=similarity,
        node_cost=node_cost,
        edge_discrepancy=discrepancy,
        source_nodes=tuple(atlas[index].roi_id for index in source.nodes),
        target_nodes=tuple(atlas[index].roi_id for index in target.nodes),
    )
