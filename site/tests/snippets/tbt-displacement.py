
import torch
from pymonntorch import Network, NeuronGroup
from conex import DisplacementCellModule, GridCellModule, TimeResolution

disp = DisplacementCellModule(
    n_cells=100,
    max_displacement=10,
    sigma=0.5,
)
grid = GridCellModule(n_modules=4, cells_per_module=25)

net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
grid_neurons = NeuronGroup(
    net=net, size=4 * 25, behavior={100: grid}, tag="grid",
)
disp_neurons = NeuronGroup(
    net=net, size=100, behavior={100: disp}, tag="disp",
)
net.initialize()

code_a = grid.encode_position(grid_neurons, torch.tensor([1, 2]))
code_b = grid.encode_position(grid_neurons, torch.tensor([4, 5]))
displacement_code = disp.compute_displacement(
    disp_neurons,
    torch.tensor([1, 2]),
    torch.tensor([4, 5]),
)
decoded = disp.decode_displacement(disp_neurons, displacement_code)

