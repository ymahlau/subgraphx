import os
import numpy as np
import torch
from collections import defaultdict

from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 200

from src.utils.utils import set_seed
from src.parameter_experiments.run_experiments import prepare_karate_experiment
from src.parameter_experiments.parameters import get_sx_params_karate
from src.utils.task_enum import Task, Experiment
from src.algorithm.subgraph_x import SubgraphX
from src.utils.metrics import sparsity, fidelity
from src.utils.logging import load_data, save_data

M=1
T=20

def compute_points():
    set_seed(0)  # IMPORTANT!!
    model, graph_loader = prepare_karate_experiment()
    sx_params = get_sx_params_karate(model, experiment=Experiment.RANDOM, max_children=-1, m=M, t=T)
    subgraphx = SubgraphX(**sx_params)

    test_graph = next(iter(graph_loader))
    test_mask = test_graph.test_mask
    test_idx = torch.arange(0, len(test_mask))[test_mask].tolist()

    # our goal is to store, for as many subgraphs as possible, tuples of (metric_score, fidelity)
    # we separate the points based on the subgraph size.
    data_points = defaultdict(list)
    save_data(f'./result_data/karate_club/metric_comparison/metric_points_m{M}t{T}', data_points)

    for node in test_idx:
        print(f'Looking at node {node} ({test_idx.index(node)+1}/{len(test_idx)})')
        nodes_to_keep = [node]

        _, mcts = subgraphx(test_graph, n_min=4, nodes_to_keep=[node], exhaustive=True)

        for mcts_node, metric_score in mcts.R.items():
            subgraph = mcts_node.node_set
            fidelity_score = fidelity(test_graph, subgraph, model, task=Task.NODE_CLASSIFICATION, nodes_to_keep=[node])
            data_points[len(subgraph)].append((metric_score, fidelity_score))

    save_data(f'./result_data/karate_club/metric_comparison/metric_points_m{M}t{T}', data_points)

def show_points():
    data_points = load_data(f'./result_data/karate_club/metric_comparison/metric_points_m{M}t{T}')

    for size, points in data_points.items():
        plt.scatter(*zip(*points), label=size, marker='.')
    plt.legend()
    plt.xlabel('Shapley Value')
    plt.ylabel('Fidelity')
    plt.show()

def compute_correlation_coefficitients(paths):
    data_points = load_data(f'./result_data/karate_club/metric_comparison/{paths[0]}')
    sizes = sorted([s for s in data_points])
    for s in sizes:
        print(f'Number of samples for size {s}: {len(data_points[s])}')
    print(f'Total: {sum([len(sam) for k,sam in data_points.items()])}')

    coefs = dict()
    for path in paths:
        coefs[path] = dict()
        data_points = load_data(f'./result_data/karate_club/metric_comparison/{path}')
        sizes = []
        for size in data_points:
            if size == 1:
                continue
            sizes.append(size)
            samples = data_points[size]
            samples = np.array(samples)
            samples = samples.T
            r = np.corrcoef(samples)[0, 1]
            if np.isnan(r):
                sizes = sizes[:-1]
                continue
            coefs[path][size] = r
            # print(f'Size {size}: Correlation coefficient for {os.path.basename(path)}: {r}')
        sizes = sorted(sizes)
        plt.plot(sizes, [coefs[path][s] for s in sizes], label=os.path.basename(path))

    plt.legend()
    plt.xlabel('subgraph size')
    plt.ylabel('correlation coefficient')
    plt.show()



def main():
    # compute_points()
    show_points()

    base_dir = './result_data/karate_club/experiments/metric_comparison'
    # names = ['metric_points_m1t5', 'metric_points_m1t20', 'metric_points_m1t100']
    names = ['metric_points_m10t5', 'metric_points_m10t20', 'metric_points_m10t100']
    paths = [os.path.join(base_dir, n) for n in names]
    compute_correlation_coefficitients(names)



if __name__ == '__main__':
    main()
