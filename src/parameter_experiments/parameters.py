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
from src.algorithm.shapley import mc_l_shapley

@dataclass
class SubgraphXSnapshot:
    """Stores a snapshot of the MCTS and explanations.

    Attributes:
        index           What this snapshot explains (e.g. node id or graph idx)
        iteration_no    Iteration of MCTS
        explanations    List of tuples (explanation-set, sparsity, fidelity)
        search_tree     Graph that contains MCTS search paths
        timestamp       Total time it took to get to this snapshot (in seconds)
    """
    index: Any
    iteration_no: int
    explanations: List[ Tuple[ Set[int], float, float] ]
    search_tree: nx.DiGraph
    timestamp: float

def get_sx_params_mutag(model, **kwargs):
    sx_params = {
        "model": model,
        "num_layers": 3,
        "exp_weight": 10,
        "m": None,
        "t": 100,
        "task": Task.GRAPH_CLASSIFICATION,
        "max_children": 12,
        "experiment": None,
        "value_func": mc_l_shapley,
    }
    for kw in kwargs:
        if kw not in sx_params:
            raise TypeError(f"Unknown Parameter: {kw}")
    sx_params.update(kwargs)

    return sx_params

def get_sx_params_karate(model, **kwargs):
    sx_params = {
        "model": model,
        "num_layers": 2,
        "exp_weight": 5,
        "m": None,
        "t": 100,
        "task": Task.NODE_CLASSIFICATION,
        "max_children": 12,
        "experiment": None,
        "value_func": mc_l_shapley,
    }
    for kw in kwargs:
        if kw not in sx_params:
            raise TypeError(f"Unknown Parameter: {kw}")
    sx_params.update(kwargs)

    return sx_params

def get_experiment_params(base_dir: str = "", **kwargs):
    experiment_params = {
        "base_dir": base_dir,
        "snapshot_after": [1, 5, 10, 15, 20],
        "n_mins": [4, 5, 6, 7, 8, 9, 10, 11, 12],
    }

    for kw in kwargs:
        if kw not in experiment_params:
            raise TypeError(f"Unknown Parameter: {kw}")
    experiment_params.update(kwargs)

    return experiment_params
