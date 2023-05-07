import networkx as nx
import torch
from dig.xgraph.dataset import SynGraphDataset
from torch.nn import ReLU, Linear, Softmax
from torch_geometric.data import Data
from torch_geometric.nn import Sequential, GCNConv, global_mean_pool

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
)
model.to(device)
print(model)

score = mc_l_shapley(model, data, {0, 1, 2}, t=100, num_layers=2)
print(score)