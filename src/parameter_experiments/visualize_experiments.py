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
plt.rcParams['figure.dpi'] = 200

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
from src.parameter_experiments.parameters import SubgraphXSnapshot


def plot_experiment(ax, data, nodes, iterations, n_mins, label, n, m, n_min, times=None):
    data = np.array(data)  # shape (nodes, iterations, n_mins)

    def is_axis(a):
        return isinstance(a, str)  # and (a[0] == 'x' or a[0] == 'y'))

    if n is None:
        data = np.mean(data, axis=0, keepdims=True)
    elif is_axis(n):
        x = nodes
    else:
        n_idx = nodes.index(n)
        data = data[[n_idx],:,:]

    if m is None:
        raise Exception("Averaging along iterations is a bad idea.")
    elif is_axis(m):
        if 'times' in m:
            x = 'times'
        else:
            x = iterations
    else:
        m_idx = iterations.index(m)
        data = data[:, [m_idx], :]

    if n_min is None:
        data = np.mean(data, axis=2, keepdims=True)
    elif is_axis(n_min):
        if 'sparsity' in n_min:
            x = 'sparsity'
        else:
            x = n_mins
    else:
        n_min_idx = n_mins.index(n_min)
        data = data[:, :, [n_min_idx]]

    data = np.reshape(data, (-1, 3))  # down to [ (fid, spar, time), (fid, spar, time), ... ]

    if x == 'sparsity':
        x = data[:, 1]
    elif x == 'times':
        x = data[:, 2]

    y = data[:, 0]  # down to [ fid, fid, fid, ... ]
    if len(y.shape) > 1:
        raise Exception("To many dimensions to plot!")

    # rearrange in ascending order
    x = np.array(x)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # now copy from vis. and also get labels as add. parameter
    ax.plot(x, y, label=label)


def reconstruct_experiment(path = None, snapshots = None):
    if snapshots is None:
        snapshots = load_data(path)

    nodes = sorted(list(snapshots.keys()))
    iterations_nos = sorted(list(snapshots[nodes[0]].keys()))
    n_mins = [len(t[0]) for t in snapshots[nodes[0]][iterations_nos[0]].explanations]
    #times = [[snapshots[nodes[n]][i].timestamp for i in iterations_nos] for n in nodes]

    data = [[[[t[2], t[1], snapshots[n][i].timestamp] for t in snapshots[n][i].explanations] for i in iterations_nos] for n in nodes]
    data = np.array(data)

    return data, nodes, iterations_nos, n_mins


def display_one_experiment(path):
    fig, ax = plt.subplots()

    snapshots = load_data(path)
    data, nodes, iterations_nos, n_mins = reconstruct_experiment(snapshots=snapshots)
    for i in iterations_nos:
        plot_experiment(ax, data, nodes, iterations_nos, n_mins, label=i,
                        n=None, m=i, n_min='x')
    ax.legend()
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Fidelity')
    plt.show()

    fig, ax = plt.subplots()

    for size in n_mins:
        plot_experiment(ax, data, nodes, iterations_nos, n_mins, label=f'{size} nodes',
                        n=None, m='x', n_min=size)
    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Fidelity')
    plt.show()

    search_tree_m = [iterations_nos[1], max([i for i in iterations_nos if i <= 20])]
    for node in nodes:
        fig, ax = plt.subplots(ncols=len(search_tree_m))
        fig.suptitle(f'node {node}')
        for i in range(len(search_tree_m)):
            m = search_tree_m[i]
            plot_search_tree(snapshots[node][m].search_tree, ax=ax[i])
            ax[i].set_title(f'iteration {m}')
        plt.show()


def compare_final_fidelity(paths: List[str], labels: List[str] = None, node: int = None, save_dst: str = None):
    if labels is None:
        labels = [os.path.basename(p) for p in paths]

    fig, ax = plt.subplots()
    for path, label in zip(paths, labels):
        snapshots = load_data(path)
        data, nodes, iterations_nos, n_mins = reconstruct_experiment(snapshots=snapshots)
        plot_experiment(ax, data, nodes, iterations_nos, n_mins, label=label,
                        n=node, m=iterations_nos[-1], n_min='x_sparsity')
    ax.legend()
    if node is None:
        ax.set_title('Average over all testing nodes')
    else:
        ax.set_title(f'Explanation for node {node}')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Fidelity')

    if save_dst is not None:
        plt.savefig(save_dst)
    plt.show()

def compare_fidelity_over_time(paths: List[str], labels: List[str] = None, node: int = None, save_dst: str = None):
    if labels is None:
        labels = [os.path.basename(p) for p in paths]

    fig, ax = plt.subplots()
    for path, label in zip(paths, labels):
        snapshots = load_data(path)
        data, nodes, iterations_nos, n_mins = reconstruct_experiment(snapshots=snapshots)
        plot_experiment(ax, data, nodes, iterations_nos, n_mins, label=label,
                        n=node, m='x_times', n_min=None)
    ax.legend()
    if node is None:
        ax.set_title('Average over all testing nodes')
    else:
        ax.set_title(f'Explanation for node {node}')
    ax.set_xlabel('Time [seconds per node]')
    ax.set_ylabel('Fidelity')

    if save_dst is not None:
        plt.savefig(save_dst)
    plt.show()


def compare_per_node(paths: List[str], labels: List[str] = None):
    snapshots = load_data(paths[0])
    nodes = list(snapshots.keys())

    for node in nodes:
        compare_final_fidelity(paths, labels, node=node)

def visualize_karate_experiments():
    """KARATE-CLUB"""
    base_dir = './result_data/karate_club/experiments'

    names = ['m20t100', 'm20t20', 'm20t5', 'm30t50', 'm20_score_fidelity',
             'greedy_shapley', 'greedy_fidelity', 'm1000_random']
    paths = [os.path.join(base_dir, n) for n in names]
    compare_final_fidelity(paths, save_dst='./img/karate_club/experiments/final_fidelity.png')
    compare_fidelity_over_time(paths, save_dst='./img/karate_club/experiments/fidelity_over_time.png')

    names = ['m30t50', 'greedy_fidelity', 'm1000_random']
    paths = [os.path.join(base_dir, n) for n in names]
    compare_per_node(paths)

    display_one_experiment(os.path.join(base_dir, 'greedy_fidelity'))


def visualize_mutag_experiments():
    """MUTAG"""
    base_dir = './result_data/mutag/experiments'

    names = ['m20t100', 'm20t5', 'greedy_one', 'm100_random']
    paths = [os.path.join(base_dir, n) for n in names]
    compare_final_fidelity(paths, save_dst='./img/mutag/experiments/final_fidelity.png')
    compare_fidelity_over_time(paths, save_dst='./img/mutag/experiments/fidelity_over_time.png')

    names = ['m20t100', 'm20t5', 'greedy_one']
    paths = [os.path.join(base_dir, n) for n in names]
    compare_per_node(paths)

    display_one_experiment(os.path.join(base_dir, 'm20t5'))

def main():
    # visualize_karate_experiments()
    visualize_mutag_experiments()


if __name__ == '__main__':
    set_seed(0)  # IMPORTANT!!
    main()
