import math
import numpy as np
from collections import defaultdict

from typing import Set, Callable, List, Dict, Union

import networkx as nx
import torch.nn
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph

from src.utils.task_enum import Task, Experiment


class MCTSNode:
    def __init__(self, graph: Data, n_min: int, node_set: Set[int], score: Union[None, float] = None):
        self.graph = graph
        self.n_min = n_min
        self.node_set = node_set
        self.score = score  # just for debugging display
        self._hash = self.compute_hash()

    def compute_hash(self):
        l = list(self.node_set)
        l.sort()
        result = 98767 - len(l) * 555
        for i, el in enumerate(l):
            result = result + (hash(el) % 9999999) * 1001 + i
        return result

    def is_terminal(self) -> bool:
        return len(self.node_set) <= self.n_min

    def __hash__(self) -> int:
        return self._hash
        #return hash(list(self.node_set).sort())

    def get_pruned_nodes(self) -> Set[int]:  # inverse of node set
        return set(range(self.graph.num_nodes)) - self.node_set

    def __str__(self) -> str:
        return f'{sorted(list(self.node_set))}: {self.score}'

    def __eq__(self, node2) -> bool:
        return hash(self) == hash(node2) and self.node_set == node2.node_set


class MCTS:
    def __init__(self, graph: Data, exp_weight: float, n_min: int, score_func: Callable, model: torch.nn.Module,
                 t: int, num_layers: int, high2low: bool = False, max_children: int = -1,
                 task: Task = Task.GRAPH_CLASSIFICATION, nodes_to_keep: List[int] = None,
                 skip_to_leaves: bool = True, experiment: Experiment = None):
        self.W = defaultdict(float)  # total reward of each node
        self.C = defaultdict(int)  # total visit count for each node
        self.children: Dict[MCTSNode, List[MCTSNode]] = {}  # nodes and their children
        self.leaves = []  # all terminal nodes
        self.R: Dict[MCTSNode, float] = {}  # immediate reward for nodes

        self.exp_weight = exp_weight  # lambda
        self.n_min = n_min
        self.score_func = score_func
        self.graph = graph
        self.model = model
        self.t = t
        self.num_layers = num_layers

        self.high2low = high2low
        self.max_children = max_children  # negative number means consider all nodes

        self.nodes_to_keep = nodes_to_keep if nodes_to_keep is not None else []
        self.task = task
        self.skip_to_leaves = skip_to_leaves  # hastens computation, but only offers explanations of size n_min
        self.experiment = experiment  # for reproducible changes in the algorithm

        if experiment == Experiment.GREEDY:
            self.C = defaultdict(lambda:1)

        if self.task == Task.GRAPH_CLASSIFICATION:
            self.root = MCTSNode(graph, n_min, set(range(graph.num_nodes)))
            self.root.score = self._r(self.root)
        else:  # for node and link classification, only consider k-hop subgraph
            node_tensor, edge_index, mapping, _ = k_hop_subgraph(nodes_to_keep, num_layers, graph.edge_index,
                                                                 relabel_nodes=False, num_nodes=graph.num_nodes,
                                                                 flow='source_to_target')
            self.root = MCTSNode(graph, n_min, set(node_tensor.tolist()))
            self.root.score = self._r(self.root)

        self.paths = []  # for visualization only

    def _q(self, mcts_node) -> float:
        if self.C[mcts_node] == 0:
            return 0  # avoid unseen moves
        return self.W[mcts_node] / self.C[mcts_node]  # average reward

    def _r(self, mcts_node) -> float:
        if mcts_node in self.R.keys():
            mcts_node.score = self.R[mcts_node]
            return self.R[mcts_node]
        else:
            if self.task == Task.GRAPH_CLASSIFICATION:
                score = self.score_func(self.model, self.graph, mcts_node.node_set, self.t, self.num_layers)
            else:  # node classification and link prediction need node indices
                score = self.score_func(self.model, self.graph, mcts_node.node_set, self.t, self.num_layers,
                                        task=self.task, nodes_to_keep=self.nodes_to_keep)
            self.R[mcts_node] = score
            mcts_node.score = score
            return score

    def _u(self, mcts_node, parent) -> float:  # utility from paper
        children = self.children[parent]
        parent_count = 0
        for c in children:
            parent_count += self.C[c]
        #counts = [self.C[n] for n in children]
        #parent_count = sum(counts)
        if parent_count == 0 and self.skip_to_leaves:  # for computational efficiency: all nodes unexplored
            return 0
        u = self.exp_weight * self._r(mcts_node) * math.sqrt(parent_count) / (1 + self.C[mcts_node])
        return u

    def _ucb(self, node, parent) -> float:  # upper confidence bound
        if self.experiment == Experiment.NO_Q:
            ucb = self._u(node, parent)
        else:
            ucb = self._q(node) + self._u(node, parent)

        return ucb

    def _select_path_by_ucb(self) -> List[MCTSNode]:  # choose best leaf by ucb, training
        mcts_node = self.root
        path = [mcts_node]
        while not mcts_node.is_terminal():
            mcts_node = self._best_child_by_ucb(mcts_node)
            path.append(mcts_node)
        return path

    def _best_child_by_ucb(self, mcts_node: MCTSNode):
        def _score_helper(child):
            return self._ucb(child, mcts_node)

        if mcts_node in self.children.keys():
            children = self.children[mcts_node]
        else:
            children = self._expand_node(mcts_node)

        # this computation matters for the random experiment
        children_scores = [_score_helper(child) for child in children]

        if self.experiment == Experiment.RANDOM:
            return np.random.choice(children)

        return max(children, key=_score_helper)

    def _expand_node(self, mcts_node) -> List[MCTSNode]:
        if mcts_node in self.children.keys():
            raise Exception(f'Node is already expanded: {mcts_node}')

        if mcts_node.is_terminal():
            raise Exception(f'terminal node cannot be expanded: {mcts_node}')

        children = []
        nx_graph = to_networkx(self.graph, to_undirected=True)  # connected components only works for undirected graphs

        # sort nodes according to pruning strategy and only consider first k nodes
        nodes_to_prune = list(mcts_node.node_set.copy())
        if self.task == Task.NODE_CLASSIFICATION or self.task == Task.LINK_PREDICTION:
            nodes_to_prune = [n for n in nodes_to_prune if n not in self.nodes_to_keep]

        nodes_to_prune.sort(key=lambda x: nx_graph.degree(x), reverse=self.high2low)
        if self.max_children >= 0:
            nodes_to_prune = nodes_to_prune[:self.max_children]

        for node in nodes_to_prune:
            subgraph = nx_graph.subgraph(mcts_node.node_set - {node})
            components = nx.connected_components(subgraph)

            if self.task == Task.GRAPH_CLASSIFICATION:  # only keep largest connected component
                child_set = max(components, key=lambda x: len(x))
            elif self.task == Task.NODE_CLASSIFICATION:  # keep component with target node
                node_to_keep = self.nodes_to_keep[0]
                child_set = set()
                for c in components:
                    if node_to_keep in c:
                        child_set = c
                        break
                if len(child_set) == 0:
                    raise Exception('Target node not in children, which should never happen. You found a bug.')

            else:  # Link prediction: keep two components which include both nodes
                node1 = self.nodes_to_keep[0]
                node2 = self.nodes_to_keep[1]
                child_set = set()
                for c in components:
                    if node1 in c or node2 in c:
                        child_set = child_set | c
                if len(child_set) == 0:
                    raise Exception('Target node not in children, which should never happen. You found a bug.')

            child = MCTSNode(self.graph, self.n_min, child_set)
            children.append(child)

        self.children[mcts_node] = children
        return children

    def _backpropagate(self, path):
        score = self._r(path[-1])  # score is reward of leaf node
        for mcts_node in path:
            self.C[mcts_node] += 1
            self.W[mcts_node] += score
        pass  # put debug breakpoint here, very useful

    def search_one_iteration(self):  # train for one iteration
        path = self._select_path_by_ucb()
        leaf = path[-1]
        if leaf not in self.leaves:
            self.leaves.append(leaf)
        self._backpropagate(path)
        self.paths.append(path)  # for later visualization

    def best_leaf_node(self) -> MCTSNode:  # choose best leaf by reward only
        return max(self.leaves, key=self._r)

    def best_node(self, size: int) -> MCTSNode:
        if self.skip_to_leaves and size != self.n_min:
            print('Warning: Some scores were skipped in the exploration phase. Set skip_to_leaves to False!')

        if size >= len(self.root.node_set):
            print('Warning: The requested explanation-set is too large.')
            return self.root
        elif size <= 0:
            raise RecursionError('There is no subgraph of the requested size')

        candidates = [k for k in self.R.keys() if len(k.node_set) == size]

        if candidates:
            return max(candidates, key=self._r)
        else:
            return self.best_node(size-1)

    def _node_info(self, mcts_node) -> str:
        pruned_nodes = list(mcts_node.get_pruned_nodes())
        pruned_nodes.sort()
        if mcts_node in self.children.keys():
            children_list = [str(child) for child in self.children[mcts_node]]
        else:
            children_list = []
        return f'-Removed {pruned_nodes}: {self._r(mcts_node):.6f}, children:{children_list}'

    def print_tree_sequential(self):  # only for debugging purposes
        print(f'mcts search tree for {self.graph}')
        print(f'Number of nodes in search tree:  {len(self.W)}')

        i = 0
        for mcts_node in self.W.keys():
            print(f'{i}: {mcts_node.info()}')
            i += 1

    def search_tree_to_networkx(self):
        def count_up():
            count_up.counter = getattr(count_up, 'counter', 0) + 1
            return count_up.counter - 1
        mcts_node_to_id = defaultdict(count_up)

        edgelist = []
        for path in self.paths:
            for i in range(len(path)-1):
                edgelist.append((mcts_node_to_id[path[i]], mcts_node_to_id[path[i+1]]))
        search_tree = nx.DiGraph(edgelist)

        node_scores = {mcts_node_to_id[mcts_node] : score for mcts_node, score in self.R.items()}
        nx.set_node_attributes(search_tree, values=node_scores, name="score")

        node_visits = {mcts_node_to_id[mcts_node] : visits for mcts_node, visits in self.C.items()}
        nx.set_node_attributes(search_tree, values=node_visits, name="visits")

        node_best = {}
        for n in range(self.n_min, len(self.root.node_set)):
            best = self.best_node(n)
            if best:
                node_best[mcts_node_to_id[best]] = True
        nx.set_node_attributes(search_tree, values=node_best, name="best")

        return search_tree

