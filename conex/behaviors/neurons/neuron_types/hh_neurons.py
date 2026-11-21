"""
Hodgkin-Huxley (1952) squid-axon neuron.

C * dv/dt = I - g_Na * m^3 * h * (v - E_Na) - g_K * n^4 * (v - E_K) - g_L * (v - E_L),
dx/dt = alpha_x * (1 - x) - beta_x * x,  for x in {m, h, n}.

Gate rates are the Hodgkin & Huxley (1952) fit at 6.3 degC in the modern
absolute-voltage convention (resting potential about -65 mV):

alpha_m = u / (1 - exp(-u)),        u = (v + 40) / 10,
beta_m  = 4 * exp(-(v + 65) / 18),
alpha_h = 0.07 * exp(-(v + 65) / 20),
beta_h  = 1 / (1 + exp(-(v + 35) / 10)),
alpha_n = 0.1 * q / (1 - exp(-q)),  q = (v + 55) / 10,
beta_n  = 0.125 * exp(-(v + 65) / 80).

All rates are in ms^-1.

References:
    Hodgkin & Huxley, "A quantitative description of membrane current
    application to conduction and excitation in nerve", J. Physiol. 1952.
    Gerstner et al., Neuronal Dynamics, Cambridge University Press, ch. 2.
"""

from pymonntorch import Behavior
import torch

# |u| below this uses the Taylor series of u / (1 - exp(-u)) at u = 0.
_EPS = 1e-6


class HH(Behavior):
    """
    Hodgkin-Huxley (1952) neuron, integrated with forward Euler.

    The input current is neurons.I in uA/cm2. Spikes are upward crossings
    of the threshold (default 0 mV); there is no reset.

    Units: v in mV, C in uF/cm2, g_Na/g_K/g_L in mS/cm2, E_Na/E_K/E_L in mV,
    neurons.I in uA/cm2, dt from the network in ms.

    Args:
        C (float): membrane capacitance in uF/cm2.
        g_Na (float): maximum sodium conductance in mS/cm2.
        g_K (float): maximum potassium conductance in mS/cm2.
        g_L (float): leak conductance in mS/cm2.
        E_Na (float): sodium reversal potential in mV.
        E_K (float): potassium reversal potential in mV.
        E_L (float): leak reversal potential in mV.
        threshold (float): spike detection threshold in mV.
        init_v (float): initial membrane potential in mV.
    """

    def __init__(
        self,
        C=1.0,
        g_Na=120.0,
        g_K=36.0,
        g_L=0.3,
        E_Na=50.0,
        E_K=-77.0,
        E_L=-54.387,
        threshold=0.0,
        *args,
        init_v=-65.0,
        **kwargs
    ):
        super().__init__(
            *args,
            C=C,
            g_Na=g_Na,
            g_K=g_K,
            g_L=g_L,
            E_Na=E_Na,
            E_K=E_K,
            E_L=E_L,
            threshold=threshold,
            init_v=init_v,
            **kwargs
        )

    def initialize(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag(self.__class__.__name__)

        neurons.C = self.parameter("C", None, required=True)
        neurons.g_Na = self.parameter("g_Na", None, required=True)
        neurons.g_K = self.parameter("g_K", None, required=True)
        neurons.g_L = self.parameter("g_L", None, required=True)
        neurons.E_Na = self.parameter("E_Na", None, required=True)
        neurons.E_K = self.parameter("E_K", None, required=True)
        neurons.E_L = self.parameter("E_L", None, required=True)
        neurons.threshold = self.parameter("threshold", None, required=True)

        neurons.v = self.parameter("init_v", neurons.vector())
        if not torch.is_tensor(neurons.v):
            neurons.v = neurons.vector(mode=neurons.v)

        # gates start at their steady state at init_v
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = self._rates(neurons)
        neurons.m = self.parameter("init_m", alpha_m / (alpha_m + beta_m))
        neurons.h = self.parameter("init_h", alpha_h / (alpha_h + beta_h))
        neurons.n = self.parameter("init_n", alpha_n / (alpha_n + beta_n))
        neurons.spikes = self.parameter("init_s", neurons.v >= neurons.threshold)

        neurons.spiking_neuron = self

    def _rates(self, neurons):
        """
        Gate rates alpha and beta in ms^-1.

        u / (1 - exp(-u)) has a removable singularity at u = 0; there the
        Taylor series 1 + u/2 + u^2/12 is used instead.
        """
        v = neurons.v

        u = (v + 40.0) / 10.0
        taylor = 1.0 + 0.5 * u + (u**2) / 12.0
        alpha_m = torch.where(u.abs() < _EPS, taylor, -u / torch.expm1(-u))
        beta_m = 4.0 * torch.exp(-(v + 65.0) / 18.0)

        alpha_h = 0.07 * torch.exp(-(v + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + torch.exp(-(v + 35.0) / 10.0))

        q = (v + 55.0) / 10.0
        taylor_n = 0.1 * (1.0 + 0.5 * q + (q**2) / 12.0)
        alpha_n = torch.where(q.abs() < _EPS, taylor_n, 0.1 * (-q / torch.expm1(-q)))
        beta_n = 0.125 * torch.exp(-(v + 65.0) / 80.0)

        return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

    def forward(self, neurons):
        """
        Single step of dynamics.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        dt = neurons.network.dt
        v = neurons.v

        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = self._rates(neurons)

        neurons.m += (alpha_m * (1.0 - neurons.m) - beta_m * neurons.m) * dt
        neurons.h += (alpha_h * (1.0 - neurons.h) - beta_h * neurons.h) * dt
        neurons.n += (alpha_n * (1.0 - neurons.n) - beta_n * neurons.n) * dt

        i_Na = neurons.g_Na * neurons.m**3 * neurons.h * (v - neurons.E_Na)
        i_K = neurons.g_K * neurons.n**4 * (v - neurons.E_K)
        i_L = neurons.g_L * (v - neurons.E_L)

        dv = (neurons.I - i_Na - i_K - i_L) * dt / neurons.C

        # spike = upward crossing of the threshold, no reset
        neurons.spikes = (v < neurons.threshold) & (v + dv >= neurons.threshold)
        v += dv
