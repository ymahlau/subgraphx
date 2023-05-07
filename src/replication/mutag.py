import math
import os
import time
from typing import Tuple, List, Dict

import networkx as nx
import torch
from dig.xgraph.dataset import MoleculeDataset
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import ReLU, Linear, Softmax, Sigmoid, Dropout
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv, GINConv, global_mean_pool, global_max_pool, GNNExplainer
from torch_geometric.utils import to_networkx

from src.algorithm.subgraph_x import SubgraphX
import src.algorithm.shapley as shapley
from src.utils.logging import load_data, save_data, aggregate_fidelity_sparsity, compute_avg_runtime, save_str
from src.utils.metrics import sparsity, fidelity
from src.utils.task_enum import Task
from src.utils.training import train_model, test
from src.utils.utils import get_device, set_seed, get_predicted_class, convert_edge_mask_to_subset
from src.utils.visualization import plot_results



def download_and_prepare_dataset() -> MoleculeDataset:
    return MoleculeDataset('./datasets', 'MUTAG')


def split_dataset(dataset: MoleculeDataset, batch_size: int = 32, split_ratio=(0.8, 0.9)) \
        -> Tuple[DataLoader, DataLoader, DataLoader, List[Data]]:
    collated_graph = dataset.data
    num_graphs = len(collated_graph.y)

    x_split_idx = dataset.slices['x'][1:-1]  # first and last index not important for split
    x_split = list(torch.tensor_split(collated_graph.x, x_split_idx, dim=0))

    edge_split_idx = dataset.slices['edge_index'][1:-1]
    edge_split = list(torch.tensor_split(collated_graph.edge_index, edge_split_idx, dim=1))

    y_tensor = collated_graph.y

    graph_list = []
    for i in range(num_graphs):
        x = x_split[i]
        edge_index = edge_split[i]
        y = y_tensor[i].long()

        graph = Data(x=x, edge_index=edge_index, y=y)
        graph_list.append(graph)

    # train, dev, test split
    neg_example_len = torch.sum(y_tensor == 0)
    pos_example_len = torch.sum(y_tensor == 1)

    all_idx = torch.arange(num_graphs)
    idx_neg = all_idx[y_tensor == 0]
    idx_pos = all_idx[y_tensor == 1]

    train_split_pos_idx = math.floor(pos_example_len * split_ratio[0])
    train_split_neg_idx = math.floor(neg_example_len * split_ratio[0])
    dev_split_pos_idx = math.floor(pos_example_len * split_ratio[1])
    dev_split_neg_idx = math.floor(neg_example_len * split_ratio[1])

    train_split_pos = idx_pos[:train_split_pos_idx]
    dev_split_pos = idx_pos[train_split_pos_idx:dev_split_pos_idx]
    test_split_pos = idx_pos[dev_split_pos_idx:]

    train_split_neg = idx_neg[:train_split_neg_idx]
    dev_split_neg = idx_neg[train_split_neg_idx:dev_split_neg_idx]
    test_split_neg = idx_neg[dev_split_neg_idx:]

    train_list = [graph_list[i] for i in [*train_split_pos, *train_split_neg]]
    dev_list = [graph_list[i] for i in [*dev_split_pos, *dev_split_neg]]
    test_list = [graph_list[i] for i in [*test_split_pos, *test_split_neg]]

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader, graph_list


def get_model_gcn():
    device = get_device()

    input_dim = 7
    hidden_dim = 128
    output_dim = 2

    model = Sequential(
        'x, edge_index, batch', [
            (GCNConv(input_dim, hidden_dim), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (global_mean_pool, 'x, batch -> x'),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(),
            Linear(hidden_dim, output_dim)
        ]
    ).to(device)

    return model


def get_model_gin():
    device = get_device()

    input_dim = 7
    hidden_dim = 128
    output_dim = 2

    model = Sequential(
        'x, edge_index, batch', [
            (GINConv(torch.nn.Sequential(Linear(input_dim, hidden_dim), ReLU(inplace=True),
                                         Linear(hidden_dim, hidden_dim), ReLU(inplace=True))), 'x, edge_index -> x'),
            (GINConv(torch.nn.Sequential(Linear(hidden_dim, hidden_dim), ReLU(inplace=True),
                                         Linear(hidden_dim, hidden_dim), ReLU(inplace=True))), 'x, edge_index -> x'),
            (GINConv(torch.nn.Sequential(Linear(hidden_dim, hidden_dim), ReLU(inplace=True),
                                         Linear(hidden_dim, hidden_dim), ReLU(inplace=True))), 'x, edge_index -> x'),
            (global_max_pool, 'x, batch -> x'),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(),
            Linear(hidden_dim, output_dim)
        ]
    ).to(device)

    return model


def train_model_or_load(train_loader, dev_loader, model_type='gcn', postfix='', add_softmax=True, only_load=False):
    save_dst = f'./checkpoints/mutag/{model_type}{postfix}.pt'
    model = get_model_gcn() if model_type == 'gcn' else get_model_gin()
    loss_func = torch.nn.CrossEntropyLoss()
    device = get_device()

    if only_load or os.path.isfile(save_dst):  # if checkpoint exists, load it
        model.load_state_dict(torch.load(save_dst))
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        num_epochs = 2000
        output_freq = 100
        train_model(model, False, optimizer, train_loader, dev_loader, num_epochs, loss_func, save_dst, output_freq)
    if add_softmax:
        # cross entropy loss already contains softmax, therefore just add softmax layer after training
        result_model = Sequential(
            'x, edge_index, batch', [
                (model, 'x, edge_index, batch -> x'),
                Softmax(dim=1),
            ]
        ).to(device)
        return result_model, loss_func
    else:
        return model, loss_func


def collect_subgraphx_expl(model, graph_list, path, workerno=None, value_func=shapley.mc_l_shapley):
    device = get_device()

    num_graphs = len(graph_list)
    test_graph_idx = torch.arange(0, num_graphs).tolist()

    if os.path.isfile(path):
        res_dict = load_data(path)
    else:
        res_dict = dict()
        for g in test_graph_idx:
            res_dict[g] = []
        save_data(path, res_dict)

    # collect explanations for all nodes with a fixed n_min
    subgraphx = SubgraphX(model, num_layers=3, exp_weight=5, m=20, t=100, task=Task.GRAPH_CLASSIFICATION,
                          value_func=value_func)

    def record_data(graph, explanation, curr_duration):
        sparsity_score = sparsity(graph, explanation)
        fidelity_score = fidelity(graph, explanation, model)

        result_tuple = (explanation, sparsity_score, fidelity_score, curr_duration)
        print(result_tuple)
        res_dict[g] = res_dict[g] + [result_tuple]

    n_mins = [4, 5, 6, 7, 8, 9, 10, 11, 12]

    for g in test_graph_idx:
        graph = graph_list[g]
        print(f'{f"worker {workerno}" if workerno is not None else ""} starting graph {g}: {graph}\n')

        start_time = time.time()
        #_, mcts = subgraphx(graph, n_min=n_mins[0], exhaustive=True)
        explanation_set,_ = subgraphx(graph, n_min=5)
        end_time = time.time()
        duration = end_time - start_time

        print(f'{f"worker {workerno}" if workerno is not None else ""} finished graph {g}')
        for n_min in [5]:#n_mins:
            if n_min > len(graph.x):
                break
            #explanation_set = mcts.best_node(n_min).node_set
            record_data(graph, explanation_set, duration)
        print()
        
    save_data(path, res_dict)
    return res_dict


def collect_gnn_expl(model, graph_list, gnnexp_epochs, path):
    device = get_device()

    num_graphs = len(graph_list)
    test_graph_idx = torch.arange(0, num_graphs).tolist()

    if os.path.isfile(path):
        res_dict = load_data(path)
    else:
        res_dict = dict()
        for g in test_graph_idx:
            res_dict[g] = []
        save_data(path, res_dict)

    # collect explanations for all nodes with a fixed n_min
    explainer = GNNExplainer(model, epochs=gnnexp_epochs, return_type='prob')

    counter = 1
    for g in test_graph_idx:
        print(f'\nstarting {counter} of {num_graphs}')
        graph = graph_list[g]
        print(graph.x.shape)
        start_time = time.time()
        node_feat_mask, edge_mask = explainer.explain_graph(graph.x.to(device), graph.edge_index.to(device))

        end_time = time.time()
        duration = end_time - start_time

        for n_min in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
            explanation_set = convert_edge_mask_to_subset(graph.edge_index, edge_mask, n_min=n_min)
            sparsity_score = sparsity(graph, explanation_set)
            fidelity_score = fidelity(graph, explanation_set, model)

            result_tuple = (explanation_set, sparsity_score, fidelity_score, duration)
            #print(result_tuple)
            res_dict[g] = res_dict[g] + [result_tuple]

        print(f'\nfinished {counter} of {num_graphs}')
        counter += 1

    save_data(path, res_dict)
    return res_dict


def runtime_study(model, graph_list, path):
    device = get_device()

    subgraphx = SubgraphX(model, num_layers=3, exp_weight=5, m=20, t=100, task=Task.GRAPH_CLASSIFICATION)
    #explainer = GNNExplainer(model, epochs=2000, return_type='prob')

    if os.path.isfile(path):
        res_dict = load_data(path)
    else:
        res_dict = dict()
        for n in range(10, 45):
            res_dict[n] = []
        save_data(path, res_dict)

    for n in range(10, 45):
        print(f"starting n={n}")
        for graph in graph_list:
            if len(graph.x) == n:
                print(f'Found graph with {n} nodes: {graph}')
                start_time = time.time()
                explanation, _ = subgraphx(graph, n_min=5)
                #node_feat_mask, edge_mask = explainer.explain_graph(graph.x.to(device), graph.edge_index.to(device))
                end_time = time.time()
                duration = end_time - start_time

                #explanation = convert_edge_mask_to_subset(graph.edge_index, edge_mask, n_min=5)

                sparsity_score = sparsity(graph, explanation)
                fidelity_score = fidelity(graph, explanation, model)

                result_tuple = (explanation, sparsity_score, fidelity_score, duration)
                print(result_tuple)
                res_dict[n] = res_dict[n] + [result_tuple]
                break
    save_data(path, res_dict)


def main():
    set_seed(0)  # IMPORTANT!
    dataset = download_and_prepare_dataset()

    batch_size = 188
    train_loader, dev_loader, test_loader, dev_list = split_dataset(dataset, batch_size, (0.8, 1.0))

    model_type = 'gin'
    print(f'Using model type \'{model_type}\'')

    model, loss_func = train_model_or_load(train_loader, dev_loader, model_type, add_softmax=True)
    test_loss, test_acc = test(model, False, dev_loader, loss_func)
    print(f'test loss: {test_loss}, test_acc: {test_acc}')
    return

    # Stats for MCTS_GNN
    path_mctsgnn = f'./result_data/mutag/{model_type}_runtime_experiment_MCTS'
    # collect_subgraphx_expl(model, dev_list, path_mctsgnn, value_func=shapley.mcts_gnn_score)
    mcts_dict = load_data(path_mctsgnn)
    mcts_sparsity, mcts_fidelity = aggregate_fidelity_sparsity(mcts_dict)
    mcts_runtime = compute_avg_runtime(mcts_dict)

    # Stats for GNNExplainer
    path_gnnexp = f'./result_data/mutag/{model_type}_runtime_experiment_GNNExp'
    # collect_gnn_expl(model, dev_list, gnnexp_epochs=2000, path=path_gnnexp)
    gnn_dict = load_data(path_gnnexp)
    gnn_sparsity, gnn_fidelity = aggregate_fidelity_sparsity(gnn_dict)
    gnn_runtime = compute_avg_runtime(gnn_dict)

    # Stats for SubgraphX
    path_subgx = f'./result_data/mutag/{model_type}_runtime_experiment_subX'
    # collect_subgraphx_expl(model, dev_list, path_subgx)
    sx_dict = load_data(path_subgx)
    sx_sparsity, sx_fidelity = aggregate_fidelity_sparsity(sx_dict)
    sx_runtime = compute_avg_runtime(sx_dict)

    # plot graph
    sparsity_list = [sx_sparsity, mcts_sparsity, gnn_sparsity]
    print(sparsity_list)
    fidelity_list = [sx_fidelity, mcts_fidelity, gnn_fidelity]
    print(fidelity_list)
    labels = ['SubgraphX', 'MCTS_GNN', 'GNNExplainer']
    plot_results(sparsity_list, fidelity_list, labels, save_dst=f'./img/mutag/{model_type}_timing_results.png')

    # save runtime to file
    data_str = f'Subgraphx: {sx_runtime}\nGNN Explainer: {gnn_runtime}\nMCTS_GNN: {mcts_runtime}'
    save_str(path=f'./result_data/mutag/{model_type}_runtime.txt', data=data_str)
    print(data_str)


if __name__ == '__main__':
    main()
