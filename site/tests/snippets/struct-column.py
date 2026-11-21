
from conex import (
    Neocortex, CorticalColumn, CorticalLayer, CorticalLayerConnection,
    PredictiveCorticalColumn,
)

net = Neocortex(dt=1.0)
# the predictive coding column builds L4, L2/3, L5, L6 and their
# feedforward / feedback / lateral connections automatically:
column = PredictiveCorticalColumn(net, column_size=100, name="col")
print(column.layer_sizes)
print(sorted(column.synapses.keys()))

