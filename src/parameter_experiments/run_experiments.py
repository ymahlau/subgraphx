import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Callable, List, Dict, Union, Tuple, Any
from enum import Enum
import pprint

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
from src.utils.task_enum import Task, Experiment
from src.utils.training import train_emb, train_model, test
from src.utils.utils import get_device, set_seed, convert_edge_mask_to_subset
from src.utils.visualization import plot_results, plot_search_tree
import src.transposition.karate_club as karate_club
import src.replication.mutag as mutag
from src.parameter_experiments.parameters import SubgraphXSnapshot,\
    get_sx_params_karate, get_sx_params_mutag, get_experiment_params
from src.algorithm.shapley import fidelity_wrapper, mcts_gnn_score


def prepare_karate_experiment():
    dataset = torch_geometric.datasets.KarateClub()
    graph = dataset.data
    device = get_device()

    emb_model = karate_club.train_or_load_embedding(graph, only_load=True)
    graph, train_loader, val_loader, test_loader = karate_club.prepare_dataset(emb_model)

    model, loss_func = karate_club.train_or_load_gcn(train_loader, val_loader, only_load=True)
    test_loss, test_acc = test(model, False, test_loader, loss_func, task=Task.NODE_CLASSIFICATION)
    print(f'test loss: {test_loss}, test_acc: {test_acc}')

    return model, test_loader

def prepare_mutag_experiment(model_type: str):
    if model_type != 'gin' and model_type != 'gcn':
        raise TypeError(f"Unknown model type: {model_type}")
    dataset = mutag.download_and_prepare_dataset()
    train_loader, dev_loader, test_loader, dev_list = mutag.split_dataset(dataset,
                                                                          batch_size=188, split_ratio=(0.8, 1.0))
    model, loss_func = mutag.train_model_or_load(train_loader, dev_loader,
                                                 model_type=model_type, add_softmax=True, only_load=True)
    test_loss, test_acc = test(model, False, dev_loader, loss_func)
    print(f'test loss: {test_loss}, test_acc: {test_acc}')

    return model, dev_list


def subgraphx_with_snapshots(sx_params, test_graph, nodes_to_keep, n_mins, snapshots, idx, snapshot_after,
                             score_and_fidelity_path = None):
    if score_and_fidelity_path:
        if os.path.isfile(score_and_fidelity_path):
            data_points = load_data(score_and_fidelity_path)
        else:
            data_points = dict()
            save_data(score_and_fidelity_path, data_points)

    subgraphx = SubgraphX(**sx_params)
    mcts = subgraphx.generate_mcts(test_graph, n_min=min(n_mins), nodes_to_keep=nodes_to_keep, exhaustive=True)
    total_time = 0
    for i in range(1, snapshot_after[-1] + 1):
        # do mcts iteration
        start_time = time.time()
        mcts.search_one_iteration()
        # explanation_set, mcts = None, None #subgraphx(test_graph, n_min=n_min, nodes_to_keep=[node])
        end_time = time.time()
        total_time += end_time - start_time

        # make snapshot if necessary
        if i in snapshot_after:
            explanations = []
            for n in n_mins:
                explanation_set = mcts.best_node(n).node_set
                sparsity_score = sparsity(test_graph, explanation_set)
                fidelity_score = fidelity(test_graph, explanation_set, sx_params["model"], task=sx_params["task"],
                                          nodes_to_keep=nodes_to_keep)
                explanation_result = (explanation_set, sparsity_score, fidelity_score)

                search_tree = mcts.search_tree_to_networkx()

                explanations.append(explanation_result)
            snap = SubgraphXSnapshot(index=idx, explanations=explanations, search_tree=search_tree,
                                     iteration_no=i, timestamp=total_time)
            snapshots[idx][i] = snap

    if score_and_fidelity_path:
        for mcts_node, metric_score in mcts.R.items():
            subgraph = mcts_node.node_set
            fidelity_score = fidelity(test_graph, subgraph, sx_params["model"], task=sx_params["task"],
                                      nodes_to_keep=nodes_to_keep)
            if len(subgraph) not in data_points:
                data_points[len(subgraph)] = []
            data_points[len(subgraph)].append((metric_score, fidelity_score))
        save_data(score_and_fidelity_path, data_points)


def run_experiment(name, experiment_params, sx_params, graph_loader, seed=0, score_samples_path=None):
    set_seed(seed)  # IMPORTANT!!
    if score_samples_path:
        score_samples_path = os.path.join(experiment_params['base_dir'], score_samples_path)

    path = os.path.join(experiment_params['base_dir'], name)
    snapshot_after = experiment_params['snapshot_after']
    n_mins = experiment_params['n_mins']
    sx_params['m'] = snapshot_after[-1]

    print('\nStarting experiment with parameters')
    print('\n'.join([f'\t{k} = {v}' for k, v in sx_params.items() if k != 'model']))
    print(f'\tsnapshot_after = {snapshot_after}')
    # print(f'\tn_mins = {n_mins}')
    print(f'\tbasename =  {os.path.basename(path)}')
    print()

    if sx_params['task'] == Task.NODE_CLASSIFICATION:
        # nodes of test set
        test_graph = next(iter(graph_loader))
        test_mask = test_graph.test_mask

        # test_idx is the list of nodes to explain
        test_idx = torch.arange(0, len(test_mask))[test_mask].tolist()
        # test_idx = [1]  #, 3]  # WE WANT SOME RESULTS, FAST
    else:
        if not isinstance(graph_loader, list):
            raise TypeError("For graph classification we do experiments given a list of graphs.")
        # test_idx is the list of graphs to explain
        num_graphs = len(graph_loader)
        test_idx = torch.arange(0, num_graphs).tolist()

    # snapshots = load_data(path)
    snapshots = dict()
    for idx in test_idx:
        snapshots[idx] = dict()

    # test saving results
    if os.path.isfile(path):
        print('Warning: overwriting old snapshots!')
        time.sleep(3)
    print('Saving works ...')
    save_data(path, snapshots)

    nodes_to_keep = None
    for idx in test_idx:
        print(f'Looking at index {idx} ({test_idx.index(idx)+1}/{len(test_idx)})')

        if sx_params['task'] == Task.NODE_CLASSIFICATION:
            nodes_to_keep = [idx]
        else:
            test_graph = graph_loader[idx]

        # do subgraphx thing
        subgraphx_with_snapshots(sx_params, test_graph, nodes_to_keep, n_mins, snapshots, idx, snapshot_after,
                                 score_and_fidelity_path=score_samples_path)

    print(f'Experiment took {sum([snaps[max(snaps.keys())].timestamp for _,snaps in snapshots.items()]):.2f} seconds')
    save_data(path, snapshots)

def karate_experiments():
    base_dir = './result_data/karate_club/experiments'
    model, graph_loader = prepare_karate_experiment()
    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1, 5, 10, 15, 20])

    # experiments with 20 iterations of mcts (same as in the paper)
    sx_params = get_sx_params_karate(model, t=100)
    run_experiment('m20t100', exp_params, sx_params, graph_loader)

    sx_params = get_sx_params_karate(model, t=20)
    run_experiment('m20t20', exp_params, sx_params, graph_loader)

    sx_params = get_sx_params_karate(model, t=5)
    run_experiment('m20t5', exp_params, sx_params, graph_loader)

    sx_params = get_sx_params_karate(model, value_func=fidelity_wrapper)
    run_experiment('m20_score_fidelity', exp_params, sx_params, graph_loader)

    sx_params = get_sx_params_karate(model, experiment=Experiment.GREEDY, t=100, max_children=-1)
    run_experiment('greedy_shapley', exp_params, sx_params, graph_loader)

    sx_params = get_sx_params_karate(model, value_func=fidelity_wrapper, experiment=Experiment.GREEDY, max_children=-1)
    run_experiment('greedy_fidelity', exp_params, sx_params, graph_loader)

    # experiments with custom iteration counts
    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1, 5, 10, 15, 20, 50, 100, 200, 500, 1000])
    sx_params = get_sx_params_karate(model, value_func=fidelity_wrapper, experiment=Experiment.RANDOM, max_children=-1)
    run_experiment('m1000_random', exp_params, sx_params, graph_loader)

    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1])
    sx_params = get_sx_params_karate(model, value_func=fidelity_wrapper, experiment=Experiment.GREEDY, max_children=-1)
    run_experiment('greedy_one', exp_params, sx_params, graph_loader)

    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1, 5, 10, 15, 20, 25, 30])
    sx_params = get_sx_params_karate(model, t=50, max_children=-1)
    run_experiment('m30t50', exp_params, sx_params, graph_loader)


def mutag_experiments(model_type: str):
    base_dir = f'./result_data/mutag/experiments_{model_type}'
    model, graph_list = prepare_mutag_experiment(model_type)
    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1, 5, 10, 15, 20])

    sx_params = get_sx_params_mutag(model)
    run_experiment('m20t100', exp_params, sx_params, graph_list)

    sx_params = get_sx_params_mutag(model, t=5)
    run_experiment('m20t5', exp_params, sx_params, graph_list)

    sx_params = get_sx_params_mutag(model, t=20)
    run_experiment('m20t20', exp_params, sx_params, graph_list)

    sx_params = get_sx_params_mutag(model, value_func=fidelity_wrapper)
    run_experiment('m20_fidelity', exp_params, sx_params, graph_list)

    sx_params = get_sx_params_mutag(model, experiment=Experiment.GREEDY, max_children=-1)
    run_experiment('m20t100_greedy', exp_params, sx_params, graph_list)

    sx_params = get_sx_params_mutag(model, value_func=fidelity_wrapper, experiment=Experiment.GREEDY, max_children=-1)
    run_experiment('m20_fidelity_greedy', exp_params, sx_params, graph_list)

    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1])
    sx_params = get_sx_params_mutag(model, value_func=fidelity_wrapper, experiment=Experiment.GREEDY, max_children=-1)
    run_experiment('m1_fidelity_greedy', exp_params, sx_params, graph_list)

    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1, 5, 10, 15, 20, 50, 100])
    sx_params = get_sx_params_mutag(model, experiment=Experiment.RANDOM, max_children=-1)
    run_experiment('m100_random', exp_params, sx_params, graph_list)

    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1, 5, 10])
    sx_params = get_sx_params_mutag(model, experiment=Experiment.RANDOM, max_children=-1)
    run_experiment('m10_shapley_random', exp_params, sx_params, graph_list, score_samples_path='random_shapley_samples')

    sx_params = get_sx_params_mutag(model, value_func=mcts_gnn_score, experiment=Experiment.RANDOM, max_children=-1)
    run_experiment('m10_mcts_gnn_random', exp_params, sx_params, graph_list, score_samples_path='random_mcts_gnn_samples')

def main():
    mutag_experiments('gin')
    mutag_experiments('gcn')
    karate_experiments()


if __name__ == '__main__':
    set_seed(0)  # IMPORTANT!!
    main()
