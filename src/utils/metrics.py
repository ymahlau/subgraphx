from typing import Set, List

import torch
from torch_geometric.data import Data

from src.utils.task_enum import Task
from src.utils.utils import get_predicted_class, get_scores, get_device


def fidelity(graph: Data, node_set: Set[int], model: torch.nn.Module, task: Task = Task.GRAPH_CLASSIFICATION,
             nodes_to_keep: List[int] = None) -> float:
    """
    Compute fidelity of an explanation
    :param graph: Graph instance to explain
    :param node_set: Explanation in form of a node set
    :param model: model to be explained
    :param task: graph, node classification or link prediction
    :param nodes_to_keep: node or link to be explained. Not necessary for graph prediction
    :return: Fidelity score
    """
    device = get_device()
    if task == Task.NODE_CLASSIFICATION or task == Task.LINK_PREDICTION:  # exclude node to explain from zero padding
        node_set = node_set - set(nodes_to_keep)

    node_tensor = torch.tensor(list(node_set)).long()

    x_occluded = torch.clone(graph.x)
    x_occluded[node_tensor] = 0

    batch = torch.zeros(graph.num_nodes).long()

    if task == Task.GRAPH_CLASSIFICATION or task == Task.NODE_CLASSIFICATION:
        scores = get_scores(model, graph.x, graph.edge_index, batch, train=False)
        scores_occluded = get_scores(model, x_occluded, graph.edge_index, batch, train=False)
    else:  # link prediction
        x1 = torch.tensor([nodes_to_keep[0]]).to(device)
        x2 = torch.tensor([nodes_to_keep[0]]).to(device)
        scores = model(graph.x.to(device), graph.edge_index.to(device), x1, x2, ptr=None).detach().cpu()
        scores_occluded = model(x_occluded.to(device), graph.edge_index.to(device), x1, x2, ptr=None).detach().cpu()

        scores = scores.squeeze()
        scores_occluded = scores_occluded.squeeze()

    if task == Task.GRAPH_CLASSIFICATION:
        predicted_class = torch.argmax(scores, dim=1)
        scores = scores[:, predicted_class]
        scores_occluded = scores_occluded[:, predicted_class]
        result = (scores - scores_occluded).item()

    elif task == Task.NODE_CLASSIFICATION:
        index_of_interest = nodes_to_keep[0]
        scores = scores[index_of_interest]
        scores_occluded = scores_occluded[index_of_interest]

        predicted_class = torch.argmax(scores, dim=0)
        scores = scores[predicted_class]
        scores_occluded = scores_occluded[predicted_class]
        result = (scores - scores_occluded).item()

    else:  # Link prediction
        predicted_class = torch.argmax(scores, dim=0)
        scores = scores[predicted_class]
        scores_occluded = scores_occluded[predicted_class]
        result = (scores - scores_occluded).item()

    return result


def sparsity(graph: Data, node_set: Set[int]) -> float:
    return 1 - (len(node_set) / graph.num_nodes)


