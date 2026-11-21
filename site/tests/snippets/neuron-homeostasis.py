from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.neurons.homeostasis import ActivityBaseHomeostasis, VoltageBaseHomeostasis
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=0.1)])
pop = NeuronGroup(net=net, size=8, behavior=[
    LIF(R=0.5, threshold=-50, tau=20,
        v_reset=-55, v_rest=-70),
    Fire(),
    ActivityBaseHomeostasis(activity_rate=1, window_size=1000,
                            updating_rate=0.2, decay_rate=1),
    # or: VoltageBaseHomeostasis(target_voltage=-60, eta_ip=0.001),
])
net.initialize()
pop.threshold = pop.vector(mode=-50)
pop.I = pop.vector(mode=40)   # constant input current (pA)
net.simulate_iterations(50000)
