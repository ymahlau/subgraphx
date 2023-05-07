import math
import os
from typing import Tuple, List

import networkx as nx
import torch
from dig.xgraph.dataset import SynGraphDataset
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import ReLU, Linear, Softmax, Sigmoid
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx

from src.algorithm.subgraph_x import SubgraphX
from src.utils.metrics import sparsity, fidelity
from src.utils.training import train_model, test
from src.utils.utils import get_device, set_seed, get_predicted_class
from src.utils.visualization import plot_search_tree


def download_and_prepare_datset() -> SynGraphDataset:
    return SynGraphDataset('./datasets', 'BA_2Motifs')


def split_dataset(dataset: SynGraphDataset, batch_size: int = 32) \
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
    idx_pos = all_idx[y_tensor == 0]
    idx_neg = all_idx[y_tensor == 1]

    train_split_pos_idx = math.floor(pos_example_len * 0.8)
    train_split_neg_idx = math.floor(neg_example_len * 0.8)
    dev_split_pos_idx = math.floor(pos_example_len * 0.9)
    dev_split_neg_idx = math.floor(neg_example_len * 0.9)

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

    return train_loader, dev_loader, test_loader, test_list


def get_model_1():
    device = get_device()

    input_dim = 10
    hidden_dim = 20
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
            Linear(hidden_dim, output_dim),
            # Sigmoid(),  # single output scalar: value between [0, 1]
            Softmax(dim=1),
        ]
    ).to(device)

    return model

def train_model_or_load_1(train_loader, dev_loader):
    save_dst = './checkpoints/ba_2motifs/gcn_temp.pt'
    model = get_model_1()
    loss_func = torch.nn.CrossEntropyLoss()

    if os.path.isfile(save_dst):  # if checkpoint exists, load it
        model.load_state_dict(torch.load(save_dst))
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        num_epochs = 800
        output_freq = 5
        train_model(model, False, optimizer, train_loader, dev_loader, num_epochs, loss_func, save_dst, output_freq)

    return model, loss_func


def load_official_model():  # TODO: this does not work yet
    save_dst = './checkpoints/ba_2motifs/gcn_latest.pth'
    device = get_device()
    loss_func = torch.nn.CrossEntropyLoss()
    input_dim = 10
    hidden_dim = 20
    output_dim = 2

    # convert official model to new pytorch version
    model = Sequential(
        'x, edge_index, batch', [
            (GCNConv(input_dim, hidden_dim, normalize=True), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(hidden_dim, hidden_dim, normalize=True), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(hidden_dim, hidden_dim, normalize=True), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (global_mean_pool, 'x, batch -> x'),
            Linear(hidden_dim, output_dim),
        ]
    ).to(device)

    # grab model weights
    net = torch.load(save_dst)['net']
    l0_w = net['model.gnn_layers.0.weight']
    l0_b = net['model.gnn_layers.0.bias']
    l1_w = net['model.gnn_layers.1.weight']
    l1_b = net['model.gnn_layers.1.bias']
    l2_w = net['model.gnn_layers.2.weight']
    l2_b = net['model.gnn_layers.2.bias']
    l3_w = net['model.mlps.0.weight']
    l3_b = net['model.mlps.0.bias']

    # copy weights into model parameters
    with torch.no_grad():  # TODO: somehow the weights get messed up in conversion, model does not perform well
        model.module_0.lin.weight.copy_(l0_w.T)
        model.module_0.bias.copy_(l0_b)
        model.module_2.lin.weight.copy_(l1_w.T)
        model.module_2.bias.copy_(l1_b)
        model.module_4.lin.weight.copy_(l2_w.T)
        model.module_4.bias.copy_(l2_b)
        model.module_7.weight.copy_(l3_w)
        model.module_7.bias.copy_(l3_b.T)

    return model, loss_func



def main():
    set_seed(1)  # IMPORTANT!
    dataset = download_and_prepare_datset()
    batch_size = 100
    train_loader, dev_loader, test_loader, test_list = split_dataset(dataset, batch_size)
    # model, loss_func = load_official_model()
    model, loss_func = train_model_or_load_1(train_loader, dev_loader)
    test_loss, test_acc = test(model, False, test_loader, loss_func)
    print(f'test loss: {test_loss}, test_acc: {test_acc}')

    # explanation
    subgraphx = SubgraphX(model, num_layers=3, exp_weight=10, m=20, t=100)

    graph = test_list[51]
    nx_graph = to_networkx(graph, to_undirected=True)
    nx.draw(nx_graph, with_labels=True)
    plt.show()
    print(graph)

    predicted_class = get_predicted_class(model, graph.x, graph.edge_index, torch.zeros(graph.num_nodes).long(),
                                          single_output=False)
    print(f'predicted: {predicted_class}')

    explanation_set, mcts = subgraphx(graph, n_min=10)

    search_tree = mcts.search_tree_to_networkx()
    name = f'search_tree_51_10'
    plot_search_tree(search_tree, f'./img/ba_2motifs/{name}.png')

    pruned_nodes = set(range(graph.num_nodes)) - explanation_set
    print(f'pruned: {pruned_nodes}')
    print(f'explanation: {explanation_set}')
    # mcts.print_tree_sequential() # only works for very small trees

    sparsity_score = sparsity(graph, explanation_set)
    print(f'sparsity: {sparsity_score}')

    fidelity_score = fidelity(graph, explanation_set, model)
    print(f'fidelity: {fidelity_score}')

    print('done')


if __name__ == '__main__':
    main()
