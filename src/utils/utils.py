import random
from typing import Set

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from src.utils.task_enum import Task


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_geometric.seed_everything(seed)


def get_scores(model: torch.nn.Module, x: torch.tensor, edge_index: torch.tensor, batch: torch.tensor,
               train: bool) -> torch.tensor:
    device = get_device()
    model.to(device)

    if train:
        scores = model(x.to(device), edge_index.to(device), batch.to(device)).detach().cpu()
    else:
        with torch.no_grad():
            scores = model(x.to(device), edge_index.to(device), batch.to(device)).detach().cpu()
    return scores


def get_predicted_class(model: torch.nn.Module, x: torch.tensor, edge_index: torch.tensor,
                        batch: torch.tensor, single_output: bool) -> torch.tensor:
    scores = get_scores(model, x, edge_index, batch, train=False)

    if single_output:
        scores = scores.squeeze()
        return torch.round(scores)
    else:
        return torch.argmax(scores, dim=1)


def convert_edge_mask_to_subset(edge_index: torch.Tensor, edge_mask: torch.Tensor, n_min: int,
                                task: Task = Task.GRAPH_CLASSIFICATION, node_to_explain: int = -1) -> Set[int]:
    edge_t = edge_index.T
    num_edges = edge_t.shape[0]

    idx_sorted = torch.argsort(edge_mask, dim=0, descending=True)

    result_set = set()
    if task == Task.NODE_CLASSIFICATION:
        result_set = {node_to_explain}  # explanation should always contain node to explain

    for i in range(num_edges):  # iteratively add nodes connected to edges to explanation set
        old_set = result_set.copy()

        cur_index = idx_sorted[i]
        edge = edge_t[cur_index]
        result_set = result_set | {edge[0].item(), edge[1].item()}

        if len(result_set) > n_min:
            return old_set

    return result_set
