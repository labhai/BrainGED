# BrainGED

This repository provides a Python implementation of BrainGED for comparing
precomputed, signed, undirected functional-connectivity graphs.

## Installation

```bash
git clone https://github.com/labhai/BrainGED.git
cd BrainGED
python -m pip install -e .
```

## Usage

```python
import numpy as np

from brainged import AtlasMetadata, NodeMetadata, brain_ged

atlas = AtlasMetadata([
    NodeMetadata(1, "L", ("Visual-A", "Visual")),
    NodeMetadata(2, "L", ("Visual-A", "Visual")),
    NodeMetadata(3, "L", ("Visual-B", "Visual")),
    NodeMetadata(4, "R", ("Visual-A", "Visual")),
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
print(result.similarity)
print(result.node_cost)
print(result.edge_discrepancy)
```

Inputs are signed, symmetric correlation matrices with the same shape and
atlas parcel order. Metadata must follow this order, with community labels
listed from fine to coarse. `top_k` specifies the number of undirected edges
retained per graph, ranked by absolute weight.

## Tests

```bash
python -m pytest -q
```

## License

This software is distributed under the [MIT License](LICENSE).
