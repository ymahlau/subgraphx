import torch
from torch.nn import ReLU, Linear, Softmax
from torch_geometric.data import Data
from torch_geometric.nn import Sequential, GCNConv, global_mean_pool

from src.algorithm.mcts import MCTS, MCTSNode

from src.algorithm.shapley import mc_l_shapley

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data = Data(x=torch.tensor([0, 1, 2, 3, 4]).unsqueeze(1).float(),
            edge_index=torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]).T.long())
print(data)

# model (untrained)
model = Sequential(
    'x, edge_index, batch', [
        (GCNConv(1, 5), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(5, 5), 'x, edge_index -> x'),
        ReLU(inplace=True),
        Linear(5, 3),
        Softmax(dim=1),
        (global_mean_pool, 'x, batch -> x')
    ]
).to(device)

n_min = 1
t = 10
score_func = mc_l_shapley
num_layers = 1

mcts = MCTS(data, exp_weight=0.5, n_min=n_min, score_func=mc_l_shapley, model=model, t=t, num_layers=num_layers)
root = MCTSNode(data, n_min, set(range(data.num_nodes)), score_func, model, t, num_layers)

mcts.search_one_iteration()

assert(len(mcts.C) == 1)
assert(len(mcts.W) == 1)
assert(mcts.C[root] == 1)

mcts.search_one_iteration()

assert(len(mcts.C) == 2)
assert(len(mcts.W) == 2)
assert(mcts.C[root] == 2)

print(mcts._best_child(root))

print('done')

