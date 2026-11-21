
import torch
from pymonntorch import Network, NeuronGroup
from conex import (
    TopDownPrediction, ErrorUnit, PrecisionWeighting,
    FreeEnergyMinimization, PredictiveCodingLearning, PredictiveCodingConfig,
    TimeResolution,
)

config = PredictiveCodingConfig(
    learning_rate=0.05,
    precision_tau=20,
    precision_learning_rate=0.01,
    min_precision=0.01,
    max_precision=100,
    error_nonlinearity="linear",
    use_sparse_errors=False,
    error_threshold=0.1,
)
behaviors = {
    230: TopDownPrediction(config=config),
    234: ErrorUnit(config=config, precision_weighted=False),
    236: PrecisionWeighting(config=config, learnable=True, attention_modulated=False),
    238: FreeEnergyMinimization(config=config, track_components=True),
    350: PredictiveCodingLearning(config=config, use_hebbian=False),
}
net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
neurons = NeuronGroup(net=net, size=32, behavior=behaviors, tag="level")
net.initialize()

torch.manual_seed(7)
sensory = torch.rand(32)            # level 0 input
higher_level = torch.rand(32)       # level 1 pattern
neurons.activity = sensory
neurons.top_down_input = higher_level
for behavior in behaviors.values():
    behavior.forward(neurons)

