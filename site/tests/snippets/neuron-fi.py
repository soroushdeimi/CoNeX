# f-I: for each current I, run the neuron and count spikes over 5000 steps.
from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF, ELIF, AELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

def fi(behavior, currents, steps=5000, dt=0.1):
    rates = []
    for I in currents:
        net = Network(behavior=[TimeResolution(dt=dt)])
        pop = NeuronGroup(net=net, size=1, behavior=[behavior, Fire()])
        net.initialize(); pop.I = pop.vector(mode=I)
        net.simulate_iterations(steps)
        rates.append(pop.spikes.sum().item() / (steps * dt / 1000))   # Hz
    return rates

currents = [20,40,60,80,100,120]   # pA
lif   = fi(LIF(R=0.5, threshold=-50, tau=20, v_reset=-55, v_rest=-70), currents)
elif_ = fi(ELIF(R=0.5, threshold=0, tau=20, v_reset=-55, v_rest=-70,
                 delta=2, theta_rh=-50), currents)
aelif = fi(AELIF(R=0.5, threshold=0, tau=20, v_reset=-55, v_rest=-70,
                 delta=2, theta_rh=-50, alpha=0, beta=60, w_tau=30), currents)
