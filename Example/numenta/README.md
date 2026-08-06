## Thousand Brains Theory Support

CoNeX implements key components from Jeff Hawkins' **Thousand Brains Theory** and Numenta's **Hierarchical Temporal Memory (HTM)** research. These components enable building brain-like systems that learn through movement, use reference frames, and reach consensus across multiple models.

### Key Components

| Component | Module | Description |
|-----------|--------|-------------|
| **Grid Cells** | `neurons.grid_cells` | Hexagonal firing patterns for allocentric (world-centered) reference frames |
| **Displacement Cells** | `neurons.grid_cells` | Encode movements between locations for learning object structure |
| **Active Dendrites** | `neurons.active_dendrites` | NMDA-based dendritic segments with nonlinear integration and contextual prediction |
| **Temporal Memory** | `neurons.sequence_memory` | HTM sequence learning algorithm with predictive cells and burst detection |
| **Spatial Pooler** | `neurons.spatial_pooler` | SDR encoding with competitive learning and homeostatic boosting |
| **SDR Operations** | `neurons.sdr` | Overlap, union, intersection, encoding, and classification for sparse patterns |
| **Column Voting** | `network.voting` | Inter-column consensus mechanism for object recognition |

### Example: Using Grid Cells

```python
from conex.behaviors.neurons import GridCellModule
from pymonntorch import Network, NeuronGroup

net = Network(behavior={}, dtype=torch.float32, device="cpu")

# Create a grid cell population with 4 modules
neurons = NeuronGroup(
    net=net,
    size=400,
    behavior={
        1: GridCellModule(
            n_modules=4,
            cells_per_module=100,
            scales=[40.0, 50.0, 70.0, 100.0],
        )
    },
    tag="grid_cells"
)
net.initialize()

# Encode a position
position = torch.tensor([50.0, 50.0])
activation = neurons.grid_cell_module.encode_position(neurons, position)
```

### Example: Using Temporal Memory

```python
from conex.behaviors.neurons import TemporalMemory, TemporalMemoryConfig

config = TemporalMemoryConfig(
    n_columns=2048,
    cells_per_column=32,
    activation_threshold=13,
)

neurons = NeuronGroup(
    net=net,
    size=2048 * 32,
    behavior={1: TemporalMemory(config=config)},
    tag="temporal_memory"
)
net.initialize()

# Process a sequence of sparse column activations
for active_columns in sequence:
    neurons.temporal_memory.compute(neurons, active_columns, learn=True)
```
