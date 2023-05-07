from typing import List, Callable

import torch.nn
from torch_geometric.data import Data

from src.algorithm.mcts import MCTS
from src.algorithm.shapley import mc_l_shapley
from src.utils.task_enum import Task, Experiment


class SubgraphX:
    def __init__(self, model: torch.nn.Module, num_layers: int, exp_weight: float, m: int, t: int,
                 high2low: bool = False, max_children: int = -1,
                 task: Task = Task.GRAPH_CLASSIFICATION,
                 value_func: Callable = mc_l_shapley, experiment: Experiment = None):
        """
        Subgraph-X Implementation from the Paper "On Explainability of Graph Neural Networks via Subgraph Explorations"
        :param model: The model to explain. Output of the model have to be normalized probabilities for each class.
        :param num_layers: The number of convolutional layers in the model
        :param exp_weight: the lambda from formula (3) in the paper. Balances exploration and exploitation
        :param m: Number of MCTS iterations
        :param t: Number of Monte-Carlo sampling steps for shapley approximation
        :param high2low: ordering of nodes when considering pruning action by their node degree
        :param max_children: Maximum number of nodes to consider for pruning actions. -1 means consider all nodes.
        :param task: Graph, Node Classification or Link prediction
        :param value_func: Function used to score subgraph explanation
        """
        self.model = model
        self.num_layers = num_layers
        self.exp_weight = exp_weight
        self.m = m
        self.t = t
        self.value_func = value_func

        self.high2low = high2low
        self.max_children = max_children

        self.task = task
        self.experiment = experiment

    def _get_mcts(self, graph: Data, n_min: int, nodes_to_keep: List[int],
                  exhaustive: bool):
        return MCTS(graph, self.exp_weight, n_min, self.value_func, self.model, self.t,
                    self.num_layers, self.high2low, self.max_children, self.task, nodes_to_keep,
                    skip_to_leaves=(not exhaustive), experiment=self.experiment)

    def __call__(self, graph: Data, n_min: int, nodes_to_keep: List[int] = None, exhaustive: bool = False):
        """
        Obtain explanation for a single instance
        :param graph: The graph to explain
        :param n_min: Maximum number of nodes in the explanation (upper bound, may not be exact)
        :param nodes_to_keep: Task dependent important nodes: None for graph classification, for node classification
        should be index of node. For link prediction both adjacent nodes to edge to explain.
        :param exhaustive: If exhaustive search is enabled, explanations of different sizes may be requested from mcts.
        :return: Set of nodes explaining the prediction of the model on the given instance
        """
        nodes_to_keep = nodes_to_keep if nodes_to_keep is not None else []
        mcts = self._get_mcts(graph, n_min, nodes_to_keep, exhaustive)

        for iteration in range(self.m):
            mcts.search_one_iteration()

        explanation = mcts.best_leaf_node()

        return explanation.node_set, mcts

    def generate_mcts(self, graph: Data, n_min: int, nodes_to_keep: List[int],
                      exhaustive: bool):
        """ Returns mcts object for in-depth experiments """
        return self._get_mcts(graph, n_min, nodes_to_keep, exhaustive)
