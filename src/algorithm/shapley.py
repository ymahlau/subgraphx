from typing import List, Set, Union, Tuple

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.utils.task_enum import Task
from src.utils.utils import get_predicted_class, get_scores, get_device

import src.utils.metrics as metrics

@torch.no_grad()
def _aggregate_scores(loader, model, class_idx, task: Task = Task.GRAPH_CLASSIFICATION,
                      nodes_to_keep: List[int] = None) -> torch.Tensor:
    device = get_device()
    result = torch.tensor([]).float()
    for data in iter(loader):
        if task == Task.GRAPH_CLASSIFICATION:
            scores = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).detach().cpu()
            score_of_class = scores[:, class_idx]
        elif task == Task.NODE_CLASSIFICATION:
            scores = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).detach().cpu()
            node_index = nodes_to_keep[0]
            idx_in_batch = data.ptr[:-1] + node_index  # ignore last pointer and find nodes in batched graphs
            score_of_node = torch.index_select(scores, dim=0, index=idx_in_batch)
            score_of_class = score_of_node[:, class_idx]  # get scores of predicted class
        else:  # link prediction
            x1 = torch.tensor([nodes_to_keep[0]]).long().to(device)  # node 1 of edge
            x2 = torch.tensor([nodes_to_keep[1]]).long().to(device)  # node 2 of edge
            scores = model(x=data.x.to(device), edge_index=data.edge_index.to(device), x1=x1, x2=x2,
                           ptr=data.ptr.to(device)).detach().cpu()
            score_of_class = scores[:, class_idx]
        result = torch.cat((result, score_of_class), dim=0)

    return result


@torch.no_grad()
def _compute_marginal_contribution(include_list, exclude_list, model, class_idx, task: Task = Task.GRAPH_CLASSIFICATION,
                                   nodes_to_keep: List[int] = None) -> float:
    include_loader = DataLoader(include_list, batch_size=4, shuffle=False, num_workers=0)
    exclude_loader = DataLoader(exclude_list, batch_size=4, shuffle=False, num_workers=0)

    include_scores = _aggregate_scores(include_loader, model, class_idx, task, nodes_to_keep)
    exclude_scores = _aggregate_scores(exclude_loader, model, class_idx, task, nodes_to_keep)

    contribution = torch.mean(include_scores - exclude_scores).item()
    return contribution


@torch.no_grad()
def mc_l_shapley(model: torch.nn.Module, graph: Data, subgraph: Set[int], t: int, num_layers: int,
                 task: Task = Task.GRAPH_CLASSIFICATION, nodes_to_keep: List[int] = None) -> float:
    """
    Shapley computation by monte carlo approximation in local neighborhood
    """
    device = get_device()
    # initialize coalition space
    subgraph_list = list(subgraph)  # v1 to vk
    if task == Task.GRAPH_CLASSIFICATION:
        node_tensor, edge_index, mapping, _ = k_hop_subgraph(subgraph_list, num_layers, graph.edge_index,
                                                             relabel_nodes=False, num_nodes=graph.num_nodes,
                                                             flow='source_to_target')  # source to target is important
    elif task == Task.NODE_CLASSIFICATION:
        node_index: int = nodes_to_keep[0]
        node_tensor, edge_index, mapping, _ = k_hop_subgraph(nodes_to_keep, num_layers, graph.edge_index,
                                                             relabel_nodes=False, num_nodes=graph.num_nodes,
                                                             flow='source_to_target')  # source to target is important
    else:
        node_tensor, edge_index, mapping, _ = k_hop_subgraph(nodes_to_keep, num_layers, graph.edge_index,
                                                             relabel_nodes=False, num_nodes=graph.num_nodes,
                                                             flow='source_to_target')  # source to target is important

    reachable_list = node_tensor.tolist()  # v1 to vr
    p_prime = list(set(reachable_list) - set(subgraph_list))

    placeholder = graph.num_nodes
    p = p_prime + [placeholder]

    if task == Task.GRAPH_CLASSIFICATION or task == Task.NODE_CLASSIFICATION:
        scores = model(x=graph.x.to(device), edge_index=graph.edge_index.to(device),
                       batch=torch.zeros(graph.num_nodes).long().to(device)).detach().cpu()
    else:
        x1 = torch.tensor([nodes_to_keep[0]]).long().to(device)  # node 1 of edge
        x2 = torch.tensor([nodes_to_keep[1]]).long().to(device)  # node 2 of edge
        scores = model(x=graph.x.to(device), edge_index=graph.edge_index.to(device), x1=x1, x2=x2, ptr=None).detach().cpu()

    if len(scores.shape) > 1:
        scores = scores.squeeze()

    if task == Task.NODE_CLASSIFICATION:
        scores = scores[node_index]  # get score for target node

    predicted_class = torch.argmax(scores, dim=0).item()

    exclude_data_list = []
    include_data_list = []

    for i in range(t):
        perm = np.random.permutation(p)
        split_idx = np.asarray(perm == placeholder).nonzero()[0][0]

        selected = perm[:split_idx]  # nodes selected for coalition

        include_mask = np.zeros(graph.num_nodes)
        include_mask[selected] = 1
        include_mask[subgraph_list] = 1  # subgraph list already includes target nodes
        masked_x_include = graph.x * torch.tensor(include_mask).unsqueeze(1)  # zero padding
        include_data = Data(masked_x_include.float(), graph.edge_index)
        include_data_list.append(include_data)

        exclude_mask = np.zeros(graph.num_nodes)
        exclude_mask[selected] = 1
        if task == Task.NODE_CLASSIFICATION or task == Task.LINK_PREDICTION:  # exclude target nodes from zero padding
            exclude_mask[nodes_to_keep] = 1
        masked_x_exclude = graph.x * torch.tensor(exclude_mask).unsqueeze(1)  # zero padding
        exclude_data = Data(masked_x_exclude.float(), graph.edge_index)
        exclude_data_list.append(exclude_data)

    score = _compute_marginal_contribution(include_data_list, exclude_data_list, model, predicted_class, task,
                                           nodes_to_keep)
    return score


@torch.no_grad()
def fidelity_wrapper(model: torch.nn.Module, graph: Data, subgraph: Set[int], t: int, num_layers: int,
                 task: Task = Task.GRAPH_CLASSIFICATION, nodes_to_keep: List[int] = None) -> float:
    """ Wrapper so we can use fidelity in place of shapley value as metric in MCTS. """
    return metrics.fidelity(graph=graph, node_set=subgraph, model=model, task=task, nodes_to_keep=nodes_to_keep)


@torch.no_grad()
def mcts_gnn_score(model: torch.nn.Module, graph: Data, subgraph: Set[int], t: int, num_layers: int,
                 task: Task = Task.GRAPH_CLASSIFICATION, nodes_to_keep: List[int] = None) -> float:
    node_set = subgraph
    device = get_device()
    # if task == Task.NODE_CLASSIFICATION or task == Task.LINK_PREDICTION:  # exclude node to explain from zero padding
    #     node_set = node_set - set(nodes_to_keep)

    node_tensor = torch.tensor(list(node_set)).long()

    # x_occluded = torch.clone(graph.x)
    # x_occluded[node_tensor] = 0

    x_subgraph = torch.zeros_like(graph.x)
    x_subgraph[node_tensor] = graph.x[node_tensor]

    batch = torch.zeros(graph.num_nodes).long()

    if task == Task.GRAPH_CLASSIFICATION or task == Task.NODE_CLASSIFICATION:
        scores = get_scores(model, graph.x, graph.edge_index, batch, train=False)
        scores_subgraph = get_scores(model, x_subgraph, graph.edge_index, batch, train=False)
    else:  # link prediction
        x1 = torch.tensor([nodes_to_keep[0]]).to(device)
        x2 = torch.tensor([nodes_to_keep[0]]).to(device)
        scores = model(graph.x.to(device), graph.edge_index.to(device), x1, x2, ptr=None).detach().cpu()
        scores_subgraph = model(x_subgraph.to(device), graph.edge_index.to(device), x1, x2, ptr=None).detach().cpu()

        scores = scores.squeeze()
        scores_subgraph = scores_subgraph.squeeze()

    if task == Task.GRAPH_CLASSIFICATION:
        predicted_class = torch.argmax(scores, dim=1)
        scores_subgraph = scores_subgraph[:, predicted_class]
        result = scores_subgraph.item()

    elif task == Task.NODE_CLASSIFICATION:
        index_of_interest = nodes_to_keep[0]
        scores = scores[index_of_interest]
        scores_subgraph = scores_subgraph[index_of_interest]

        predicted_class = torch.argmax(scores, dim=0)
        scores_subgraph = scores_subgraph[predicted_class]
        result = scores_subgraph.item()

    else:  # Link prediction
        predicted_class = torch.argmax(scores, dim=0)
        scores_subgraph = scores_subgraph[predicted_class]
        result = scores_subgraph.item()

    return result
