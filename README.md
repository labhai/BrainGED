# BrainGED

Minimal reference implementation of BrainGED, a hierarchical community-aware
similarity measure for sparsified functional brain networks.

## Installation

```bash
python -m pip install -e .
```

## Usage

```python
import numpy as np

from brainged import AtlasMetadata, NodeMetadata, brain_ged

atlas = AtlasMetadata([
    NodeMetadata(1, "L", "Visual-A", "Visual"),
    NodeMetadata(2, "L", "Visual-A", "Visual"),
    NodeMetadata(3, "L", "Visual-B", "Visual"),
    NodeMetadata(4, "R", "Visual-A", "Visual"),
])

rest = np.array([
    [0.0,  0.8, -0.2, 0.1],
    [0.8,  0.0,  0.6, 0.0],
    [-0.2, 0.6,  0.0, 0.3],
    [0.1,  0.0,  0.3, 0.0],
])
task = np.array([
    [0.0,  0.7, -0.5, 0.1],
    [0.7,  0.0,  0.2, 0.0],
    [-0.5, 0.2,  0.0, 0.4],
    [0.1,  0.0,  0.4, 0.0],
])

result = brain_ged(rest, task, atlas, top_k=2)
print(result.similarity, result.node_cost, result.edge_discrepancy)
```

The two inputs must be dense, signed, symmetric Pearson FC matrices in the same
atlas parcel order. BrainGED retains the global top-`k` upper-triangle edges
ranked by absolute weight, preserves their signs, and removes isolated nodes
for the node edit term. Exact ties are resolved by atlas index. Node
substitutions are restricted to the same hemisphere and Yeo-7 network. The
edge term is evaluated separately in the unchanged full-atlas parcel order.

The returned score is

```text
similarity = 1 - (node_cost + edge_discrepancy) / 2
```

The implementation uses deletion and insertion costs of `1`, the hierarchy
`parcel identity -> Yeo-17 -> Yeo-7`, and `1e-12` for both numerical constants.

Run the tests with:

```bash
python -m pytest -q
```

Citation information will be added when the associated paper is public.

## License

MIT
