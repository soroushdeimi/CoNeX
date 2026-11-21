from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=0.1)])
pop = NeuronGroup(net=net, size=1, behavior=[
    LIF(R=0.5, threshold=-50, tau=20,
        v_reset=-55, v_rest=-70),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=65)   # constant input current (pA)
net.simulate_iterations(500)
