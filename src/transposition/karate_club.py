import os
import random
import time
from collections import defaultdict
from typing import Dict

import networkx as nx
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch_geometric.datasets
from torch import optim
from torch.nn import ReLU, Linear, Softmax
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Sequential, GNNExplainer

from src.algorithm.subgraph_x import SubgraphX
from src.utils.logging import load_data, save_data, aggregate_fidelity_sparsity, compute_avg_runtime, save_str
from src.utils.metrics import sparsity, fidelity
from src.utils.task_enum import Task
from src.utils.training import train_emb, train_model, test
from src.utils.utils import get_device, set_seed, convert_edge_mask_to_subset
from src.utils.visualization import plot_results, plot_search_tree

batch_size = 1  # there is only a single graph in the dataset
num_classes = 4
num_nodes = 34

def get_model():
    device = get_device()

    input_dim = 34
    hidden_dim = 20
    output_dim = 4

    model = Sequential(
        'x, edge_index, batch', [
            (GCNConv(input_dim, hidden_dim), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(hidden_dim, output_dim), 'x, edge_index -> x'),
            Softmax(dim=1),
        ]
    ).to(device)

    return model

def prepare_dataset(emb_model):
    dataset = torch_geometric.datasets.KarateClub()
    graph = dataset.data
    train_mask = graph.train_mask
    y = graph.y

    """
    distribution of nodes in train, test, val:
    class 0: 1 train, 6 val, 6 test
    class 1: 1 train, 6 val, 5 test
    class 2: 1 train, 2 val, 1 test
    class 3: 1 train, 2 val, 2 test
    """

    val_0_idx = random.sample(range(12), k=6)
    test_0_idx = list(set(range(12)) - set(val_0_idx))
    val_1_idx = random.sample(range(11), k=6)
    test_1_idx = list(set(range(11)) - set(val_1_idx))
    val_2_idx = random.sample(range(3), k=2)
    test_2_idx = list(set(range(3)) - set(val_2_idx))
    val_3_idx = random.sample(range(4), k=2)
    test_3_idx = list(set(range(4)) - set(val_3_idx))

    val_indices = [val_0_idx, val_1_idx, val_2_idx, val_3_idx]
    test_indices = [test_0_idx, test_1_idx, test_2_idx, test_3_idx]

    val_mask = torch.zeros_like(train_mask).bool()
    test_mask = torch.zeros_like(train_mask).bool()

    for class_i in range(num_classes):
        counter = 0  # the number of nodes of this class already processed
        cur_val_idx = val_indices[class_i]
        cur_test_idx = test_indices[class_i]

        for node in range(num_nodes):  # assign nodes to test and val mask
            node_class = y[node].item()
            if node_class == class_i and not train_mask[node]:
                # node is either in test or val mask
                if counter in cur_val_idx:
                    val_mask[node] = True
                else:
                    test_mask[node] = True
                counter += 1

    new_x = emb_model.forward().data.cpu().detach()

    new_graph = Data(x=new_x, edge_index=graph.edge_index, y=y, train_mask=train_mask, val_mask=val_mask,
                     test_mask=test_mask)
    train_loader = DataLoader([new_graph], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([new_graph], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([new_graph], batch_size=batch_size, shuffle=False)
    return new_graph, train_loader, val_loader, test_loader

def get_embedding_model(graph):
    device = get_device()
    embedding_dim = 10
    walk_length = 80
    context_size = 10
    walks_per_node = 10

    model = torch_geometric.nn.Node2Vec(graph.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
                                        context_size=context_size, walks_per_node=walks_per_node,
                                        num_nodes=num_nodes).to(device)
    loader = model.loader(num_workers=0, batch_size=64, shuffle=True)
    return model, loader

def train_or_load_embedding(graph, only_load=False):
    save_path = './checkpoints/karate_club/emb'
    emb_model, emb_loader = get_embedding_model(graph)

    if only_load or os.path.isfile(save_path):  # if checkpoint exists, load it
        emb_model.load_state_dict(torch.load(save_path))
    else:
        emb_optimizer = Adam(emb_model.parameters())
        num_epochs = 2000
        train_emb(emb_model, emb_optimizer, emb_loader, num_epochs, output_freq=100)
        torch.save(emb_model.state_dict(), save_path)
    return emb_model

def get_gcn_model():
    device = get_device()
    input_dim = 10
    hidden_dim = 20
    output_dim = 4

    model = Sequential(
        'x, edge_index, batch', [
            (GCNConv(input_dim, hidden_dim), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(hidden_dim, output_dim), 'x, edge_index -> x'),
            # Softmax(dim=1),
        ]
    ).to(device)

    return model

def train_or_load_gcn(train_loader, val_loader, only_load=False):
    save_path = './checkpoints/karate_club/gcn.pt'
    model = get_gcn_model()
    loss_func = torch.nn.CrossEntropyLoss()
    device = get_device()

    if only_load or os.path.isfile(save_path):
        model.load_state_dict(torch.load(save_path))
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 1000
        output_freq = 50
        train_model(model, False, optimizer, train_loader, val_loader, num_epochs, loss_func, save_path, output_freq,
                    task=Task.NODE_CLASSIFICATION)

    # cross entropy loss already contains softmax, therefore just add softmax layer after training
    result_model = Sequential(
        'x, edge_index, batch', [
            (model, 'x, edge_index, batch -> x'),
            Softmax(dim=1),
        ]
    ).to(device)

    # model.add_module('module_3', Softmax(dim=1))

    return result_model, loss_func

def debug(model, test_loader):
    device = get_device()
    # get scores for test set
    test_graph = next(iter(test_loader))
    test_mask = test_graph.test_mask

    scores = model(test_graph.x.to(device), test_graph.edge_index.to(device),
                   test_graph.batch.to(device)).detach().cpu()
    pred = torch.argmax(scores, dim=1)

    test_node_idx = torch.arange(0, num_nodes)[test_mask]
    test_scores = scores[test_mask]
    test_pred = pred[test_mask]
    truth = test_graph.y[test_mask]

    node = test_node_idx[0].item()
    print(f'testing explanation for node {node}')
    subgraphx = SubgraphX(model, num_layers=2, exp_weight=5, m=50, t=50, task=Task.NODE_CLASSIFICATION)
    explanation_set, mcts = subgraphx(test_graph, n_min=10, nodes_to_keep=[node])
    print(f'explanation: {explanation_set}')

    sparsity_score = sparsity(test_graph, explanation_set)
    print(f'sparsity: {sparsity_score}')

    fidelity_score = fidelity(test_graph, explanation_set, model, task=Task.NODE_CLASSIFICATION, nodes_to_keep=[node])
    print(f'fidelity: {fidelity_score}')


def debug_2(model, test_loader):
    device = get_device()
    # get scores for test set
    test_graph = next(iter(test_loader))
    test_mask = test_graph.test_mask

    scores = model(test_graph.x.to(device), test_graph.edge_index.to(device),
                   test_graph.batch.to(device)).detach().cpu()
    pred = torch.argmax(scores, dim=1)

    test_node_idx = torch.arange(0, num_nodes)[test_mask]
    test_scores = scores[test_mask]
    test_pred = pred[test_mask]
    truth = test_graph.y[test_mask]

    node = test_node_idx[0].item()
    print(f'testing explanation for node {node}')

    explainer = GNNExplainer(model, epochs=200, return_type='prob')
    batch = torch.zeros(num_nodes).to(device)
    node_feat_mask, edge_mask = explainer.explain_node(node, test_graph.x.to(device), test_graph.edge_index.to(device),
                                                       batch=batch)

    ax, G = explainer.visualize_subgraph(node, test_graph.edge_index, edge_mask, y=test_graph.y)
    plt.show()

    explanation_set = convert_edge_mask_to_subset(test_graph.edge_index, edge_mask, n_min=10,
                                                  task=Task.NODE_CLASSIFICATION, node_to_explain=node)

    print(f'explanation: {explanation_set}')

    sparsity_score = sparsity(test_graph, explanation_set)
    print(f'sparsity: {sparsity_score}')

    fidelity_score = fidelity(test_graph, explanation_set, model, task=Task.NODE_CLASSIFICATION, nodes_to_keep=[node])
    print(f'fidelity: {fidelity_score}')

"""
For every node in the test set multiple data points are collected. Each point consists of: 
(1) explanation node set
(2) sparsity
(3) fidelity
(4) time to compute explanation (seconds)
"""
def collect_subgraphx_expl(model, test_loader, only_one_mcts = False):
    device = get_device()

    # nodes of test set
    test_graph = next(iter(test_loader))
    test_mask = test_graph.test_mask
    test_node_idx = torch.arange(0, num_nodes)[test_mask].tolist()

    path = './result_data/karate_club/subgraphx_new_dict3'
    if only_one_mcts:
        path = f'{path}_once'
    if os.path.isfile(path):
        res_dict = load_data(path)
    else:
        res_dict = dict()
        for node in test_node_idx:
            res_dict[node] = []
        # save_data(path, res_dict)

    # collect explanations for all nodes with a fixed n_min
    subgraphx = SubgraphX(model, num_layers=2, exp_weight=5, m=30, t=50, task=Task.NODE_CLASSIFICATION)

    def record_data(node, explanation, duration):
        sparsity_score = sparsity(test_graph, explanation_set)
        fidelity_score = fidelity(test_graph, explanation_set, model, task=Task.NODE_CLASSIFICATION,
                                  nodes_to_keep=[node])

        result_tuple = (explanation_set, sparsity_score, fidelity_score, duration)
        print(result_tuple)
        res_dict[node] = res_dict[node] + [result_tuple]

    n_mins = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    if not only_one_mcts:
        for n_min in n_mins:
            counter = 1
            print(f'\nstarting {n_min}')
            for node in test_node_idx:
                start_time = time.time()
                explanation_set, _ = subgraphx(test_graph, n_min=n_min, nodes_to_keep=[node], exhaustive=False)
                end_time = time.time()
                duration = end_time - start_time

                record_data(node, explanation_set, duration)
                print(f'finished node {counter} of {len(test_node_idx)}')
                counter += 1
    else:
        for node in test_node_idx:
            print(f'starting node {node}')
            start_time = time.time()
            _, mcts = subgraphx(test_graph, n_min=n_mins[0], nodes_to_keep=[node], exhaustive=True)
            end_time = time.time()
            duration = end_time - start_time

            for n_min in n_mins:
                explanation_set = mcts.best_node(n_min).node_set
                record_data(node, explanation_set, duration)

    # save_data(path, res_dict)
    return res_dict


def collect_gnn_expl(model, test_loader) -> Dict:
    device = get_device()

    # nodes of test set
    test_graph = next(iter(test_loader))
    test_mask = test_graph.test_mask
    test_node_idx = torch.arange(0, num_nodes)[test_mask].tolist()

    path = './result_data/karate_club/ggn_exp_dict'
    if os.path.isfile(path):
        res_dict = load_data(path)
    else:
        res_dict = dict()
        for node in test_node_idx:
            res_dict[node] = []
        # save_data(path, res_dict)

    # collect explanations for all nodes with a fixed n_min
    explainer = GNNExplainer(model, epochs=2000, return_type='prob')

    counter = 1
    for node in test_node_idx:
        print(f'\nstarting {counter} of {len(test_node_idx)}')
        start_time = time.time()

        batch = torch.zeros(num_nodes).to(device)
        node_feat_mask, edge_mask = explainer.explain_node(node, test_graph.x.to(device),
                                                           test_graph.edge_index.to(device),
                                                           batch=batch)
        end_time = time.time()
        duration = end_time - start_time

        for n_min in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
            explanation_set = convert_edge_mask_to_subset(test_graph.edge_index, edge_mask, n_min=n_min,
                                                          task=Task.NODE_CLASSIFICATION, node_to_explain=node)
            sparsity_score = sparsity(test_graph, explanation_set)
            fidelity_score = fidelity(test_graph, explanation_set, model, task=Task.NODE_CLASSIFICATION,
                                      nodes_to_keep=[node])

            result_tuple = (explanation_set, sparsity_score, fidelity_score, duration)
            res_dict[node] = res_dict[node] + [result_tuple]
        print(f'finished {counter} of {len(test_node_idx)}')
        counter += 1

    # save_data(path, res_dict)
    return res_dict

def explain_one(model, test_loader, node, load=False):
    name = f'search_tree_{node}'

    if not load:
        device = get_device()

        # nodes of test set
        test_graph = next(iter(test_loader))

        # collect explanations for all nodes with a fixed n_min
        subgraphx = SubgraphX(model, num_layers=2, exp_weight=5, m=20, t=20, task=Task.NODE_CLASSIFICATION)
        n_min = 4

        start_time = time.time()
        explanation_set, mcts = subgraphx(test_graph, n_min=n_min, nodes_to_keep=[node])
        end_time = time.time()
        duration = end_time - start_time

        search_tree = mcts.search_tree_to_networkx()
        save_data(f'./result_data/karate_club/{name}', search_tree)

        sparsity_score = sparsity(test_graph, explanation_set)
        fidelity_score = fidelity(test_graph, explanation_set, model, task=Task.NODE_CLASSIFICATION,
                                          nodes_to_keep=[node])

        result_tuple = (explanation_set, sparsity_score, fidelity_score, duration)
    else:
        search_tree = load_data(f'./result_data/karate_club/{name}')
        result_tuple = None

    plot_search_tree(search_tree, f'./img/karate_club/{name}.png')

    return result_tuple


def main():
    dataset = torch_geometric.datasets.KarateClub()
    graph = dataset.data
    device = get_device()

    # print graph
    # color_map = []
    # for node in range(num_nodes):
    #     if graph.y[node].item() == 0:
    #         color_map.append('red')
    #     elif graph.y[node].item() == 1:
    #         color_map.append('blue')
    #     elif graph.y[node].item() == 2:
    #         color_map.append('green')
    #     elif graph.y[node].item() == 3:
    #         color_map.append('yellow')
    # nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    # nx.draw(nx_graph, node_color=color_map, with_labels=True)
    # plt.savefig('./img/karate_club/graph.png')
    # plt.show()

    # first train embedding
    emb_model = train_or_load_embedding(graph)
    graph, train_loader, val_loader, test_loader = prepare_dataset(emb_model)

    # then train gcn
    model, loss_func = train_or_load_gcn(train_loader, val_loader)
    test_loss, test_acc = test(model, False, test_loader, loss_func, task=Task.NODE_CLASSIFICATION)
    print(f'test loss: {test_loss}, test_acc: {test_acc}')

    # debug both explanation methods
    # debug(model, test_loader)
    # debug_2(model, test_loader)
    result = explain_one(model=model, test_loader=test_loader, node=12)
    print(result)
    return

    # collect_subgraphx_expl(model, test_loader, only_one_mcts=True)
    # collect_gnn_expl(model, test_loader)

    sx_dict = load_data('./result_data/karate_club/subgraphx_new_dict2')
    sx_sparsity, sx_fidelity = aggregate_fidelity_sparsity(sx_dict)
    sx_runtime = compute_avg_runtime(sx_dict)

    gnn_dict = load_data('./result_data/karate_club/ggn_exp_dict')
    gnn_sparsity, gnn_fidelity = aggregate_fidelity_sparsity(gnn_dict)
    gnn_runtime = compute_avg_runtime(gnn_dict)

    sx_once_dict = load_data('./result_data/karate_club/subgraphx_new_dict3_once')
    sx_once_sparsity, sx_once_fidelity = aggregate_fidelity_sparsity(sx_once_dict)
    sx_once_runtime = compute_avg_runtime(sx_once_dict)

    # plot graph
    sparsity_list = [sx_sparsity, gnn_sparsity, sx_once_sparsity]
    fidelity_list = [sx_fidelity, gnn_fidelity, sx_once_fidelity]
    labels = ['SubgraphX', 'GNN Explainer', 'SubgraphX_N4']
    plot_results(sparsity_list, fidelity_list, labels, save_dst='./img/karate_club/karate_results_once_3.png')

    # save runtime to file
    data_str = f'Subgraphx: {sx_runtime}\nGNN Explainer: {gnn_runtime}\nSubgraphX_N4: {sx_once_runtime}'
    save_str(path='./result_data/karate_club/runtime.txt', data=data_str)
    print(data_str)


if __name__ == '__main__':
    set_seed(0)  # IMPORTANT!!
    main()
