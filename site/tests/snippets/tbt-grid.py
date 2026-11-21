
import torch
from pymonntorch import Network, NeuronGroup
from conex import GridCellModule, TimeResolution

module = GridCellModule(
    n_modules=4,
    cells_per_module=25,
    scale_range=(0.5, 2),
    orientation_range=(0.0, 1.047),
)
net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
neurons = NeuronGroup(
    net=net, size=4 * 25, behavior={100: module}, tag="grid",
)
net.initialize()

# path integration: position += velocity * dt
module.path_integrate(neurons, torch.tensor([1.0, 0.0]), dt=0.05)
activation = module.encode_position(neurons, torch.tensor([1.0, 1.0]))

