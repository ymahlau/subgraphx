from typing import List, Union

from matplotlib import pyplot as plt

import networkx as nx
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout

def plot_results(sparsity_list: List[List[float]], fidelity_list: List[List[float]], labels: List[str],
                 save_dst: Union[None, str] = None):
    for sparsity, fidelity, label in zip(sparsity_list, fidelity_list, labels):
        plt.plot(sparsity, fidelity, label=label)
    plt.legend()
    plt.xlabel('Sparsity')
    plt.ylabel('Fidelity')

    if save_dst is not None:
        plt.savefig(save_dst)

    plt.show()

def plot_search_tree(search_tree: nx.DiGraph, save_dst: str = None, ax=None):
    labels = nx.get_node_attributes(search_tree, 'score')
    labels = {n: float(f'{s:.2g}') for n, s in labels.items()}

    def get_color(n):
        color_map = {0: 'black', 1: 'blue', 2: 'green', 3: 'yellow'}
        if n in color_map:
            return color_map[n]
        else:
            return 'red'

    colors = nx.get_node_attributes(search_tree, 'visits')
    colors = {n: get_color(s) for n, s in colors.items()}
    colors = [c for n, c in sorted(colors.items())]

    best = nx.get_node_attributes(search_tree, 'best')
    borders = colors.copy()
    for n, is_best in best.items():
        if is_best:
            borders[n] = 'red'

    pos = graphviz_layout(search_tree, prog="dot")

    if ax is None:
        fig, ax = plt.subplots(dpi=200)
        nx.draw(search_tree, pos, ax=ax, node_size=50, font_size=4, labels=labels, node_color=colors,
                edgecolors=borders)
                #linewidths=borders, edgecolors=borders)

        if save_dst:
            fig.savefig(save_dst)

        plt.show()
    else:
        nx.draw(search_tree, pos, ax=ax, node_size=50, font_size=4, labels=labels, node_color=colors,
                edgecolors=borders)
