
import torch
from pymonntorch import Network, NeuronGroup
from conex import DendriticSegment, DendriticSegmentConfig, ContextualPrediction, TimeResolution

config = DendriticSegmentConfig(
    n_synapses_per_segment=6,
    activation_threshold=6,
    learning_threshold=4,
    connected_permanence=0.2,
)
segments = DendriticSegment(
    n_segments=4, n_cells=12, config=config,
)
context = ContextualPrediction(depolarization_threshold=0.5)

net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
neurons = NeuronGroup(
    net=net, size=12, behavior={100: segments, 250: context}, tag="segments",
)
net.initialize()

# wire segment 0 to a few presynaptic cells, then present a context SDR
for slot, target in enumerate([1, 2, 3]):
    neurons.segment_targets[0, slot] = target
    neurons.segment_permanences[0, slot] = 0.9
presynaptic_activity = torch.zeros(12, dtype=torch.bool)
presynaptic_activity[[1, 2, 3]] = True

segments.compute_segment_activation(neurons, presynaptic_activity)
segments.compute_depolarization(neurons)
context.update_predictions(neurons, neurons.depolarization)

