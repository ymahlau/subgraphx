import torch
print(torch.cuda.is_available())

import torch_geometric
print(torch_geometric.__version__)

import dig
from dig.xgraph.method import SubgraphX
print('all installations complete')

