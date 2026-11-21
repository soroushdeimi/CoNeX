
import torch
from pymonntorch import Network, NeuronGroup
from conex import ColumnVoting, ConsensusNetwork, TimeResolution

voting = ColumnVoting(
    n_objects=6,
    representation_size=64,
    sparsity=0.25,
    voting_threshold=0.5,
    decay_rate=0.1,
    min_votes_for_consensus=3,
)
consensus = ConsensusNetwork(convergence_threshold=0.8)

net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
columns = [
    NeuronGroup(net=net, size=64, behavior={100: voting}, tag=f"col{c}")
    for c in range(3)
]
net.initialize()
consensus.initialize(net)

# object memories: fixed random SDRs; local rep = object 0 with a few bits flipped
torch.manual_seed(7)
n_active = max(1, int(64 * 0.25))
object_memories = torch.zeros(6, 64)
for o in range(6):
    object_memories[o, torch.randperm(64)[:n_active]] = 1.0
local_representation = object_memories[0].clone()
local_representation[:2] = 0.0
local_representation[-2:] = 1.0

votes = voting.cast_vote(columns[0], local_representation, object_memories)
voting.receive_votes(columns[0], votes)
consensus.broadcast_votes(net, columns)
voting.check_consensus(columns[0], n_columns=3)

