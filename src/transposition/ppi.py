import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch_geometric
from tdc.multi_pred import PPI
from torch import optim, Tensor
from torch.nn import ReLU, Softmax
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.utils import to_networkx, to_undirected
from torch_geometric.data import Data

from src.algorithm.subgraph_x import SubgraphX
from src.utils.logging import load_data, save_data, aggregate_fidelity_sparsity, compute_avg_runtime, save_str
from src.utils.metrics import sparsity, fidelity
from src.utils.task_enum import Task
from src.utils.training import train_emb, train_model, test
from src.utils.utils import set_seed, get_device
from src.utils.visualization import plot_results

num_nodes = 8248
embedding_dim = 128


def prepare_dataset(seed):
    # download data
    ppi = 'HuRI'
    ppi_data = PPI(name=ppi, path='./datasets/ppi')
    neg_data = ppi_data.neg_sample(frac=1)
    # This would return a pytorch-geometric graph
    graph_dic = ppi_data.to_graph(format='pyg', split=True, seed=seed)

    # mapping function
    def id_to_int_func(node_id):
        id_to_int = graph_dic['index_to_entities']
        return id_to_int[node_id]

    # train split
    train_split_table = graph_dic['split']['train']
    id1_list_train = train_split_table['Protein1_ID'].values
    id2_list_train = train_split_table['Protein2_ID'].values
    node1_list_train = list(map(id_to_int_func, id1_list_train))
    node2_list_train = list(map(id_to_int_func, id2_list_train))
    edge_list_train = torch.zeros((2, len(node1_list_train))).long()
    edge_list_train[0] = torch.tensor(node1_list_train)
    edge_list_train[1] = torch.tensor(node2_list_train)
    y_train = torch.tensor(train_split_table['Y'].values).long()

    # val split
    val_split_table = graph_dic['split']['valid']
    id1_list_val = val_split_table['Protein1_ID'].values
    id2_list_val = val_split_table['Protein2_ID'].values
    node1_list_val = list(map(id_to_int_func, id1_list_val))
    node2_list_val = list(map(id_to_int_func, id2_list_val))
    edge_list_val = torch.zeros((2, len(node1_list_val))).long()
    edge_list_val[0] = torch.tensor(node1_list_val)
    edge_list_val[1] = torch.tensor(node2_list_val)
    y_val = torch.tensor(val_split_table['Y'].values).long()

    # test split
    test_split_table = graph_dic['split']['test']
    id1_list_test = test_split_table['Protein1_ID'].values
    id2_list_test = test_split_table['Protein2_ID'].values
    node1_list_test = list(map(id_to_int_func, id1_list_test))
    node2_list_test = list(map(id_to_int_func, id2_list_test))
    edge_list_test = torch.zeros((2, len(node1_list_test))).long()
    edge_list_test[0] = torch.tensor(node1_list_test)
    edge_list_test[1] = torch.tensor(node2_list_test)
    y_test = torch.tensor(test_split_table['Y'].values).long()

    # graph from positive training edges
    edge_list_1 = np.array(node1_list_train)[y_train == 1]
    edge_list_2 = np.array(node2_list_train)[y_train == 1]
    edge_index = torch.zeros((2, len(edge_list_1)))
    edge_index[0] = torch.tensor(edge_list_1)
    edge_index[1] = torch.tensor(edge_list_2)
    edge_index = edge_index.long()
    edge_index = to_undirected(edge_index)  # convert positive training edges to undirected
    graph = Data(torch.nn.functional.one_hot(x=torch.arange(num_nodes)).float(), edge_index=edge_index)
    # graph = Data(x=torch.arange(num_nodes).long(), edge_index=edge_index)

    edge_lists = [edge_list_train, edge_list_val, edge_list_test]
    y_lists = [y_train, y_val, y_test]

    return edge_lists, y_lists, graph


def train_or_load_embedding(train_edge_index):
    # train or load unsupervised embedding
    device = get_device()
    walk_length = 80
    context_size = 10
    walks_per_node = 10
    save_path = './checkpoints/ppi/emb2.model'
    batch_size = 64
    emb_model = torch_geometric.nn.Node2Vec(train_edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
                                            context_size=context_size, walks_per_node=walks_per_node,
                                            num_nodes=num_nodes).to(device)
    loader = emb_model.loader(num_workers=0, batch_size=batch_size, shuffle=True)

    if os.path.isfile(save_path):  # if checkpoint exists, load it
        print('loading local embedding model')
        emb_model.load_state_dict(torch.load(save_path))
    else:
        emb_optimizer = Adam(emb_model.parameters())
        num_epochs = 50
        train_emb(emb_model, emb_optimizer, loader, num_epochs, output_freq=1)
        torch.save(emb_model.state_dict(), save_path)

    return emb_model


def update_graph_features(emb_model):
    device = get_device()
    x = torch.arange(num_nodes).long()
    new_x = emb_model.forward(x.to(device)).detach().cpu()
    return new_x


class LinkPredictorModel(torch.nn.Module):
    def __init__(self):
        super(LinkPredictorModel, self).__init__()
        self.hidden_dim = 64
        self.gcn1 = GCNConv(embedding_dim, self.hidden_dim, cached=False)
        # self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim, cached=True)
        # self.lin1 = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.lin2 = torch.nn.Linear(2 * self.hidden_dim, 2)

    def forward(self, x, edge_index, node1, node2, ptr: Optional[Tensor] = None):
        x = self.gcn1(x, edge_index)
        x = torch.nn.functional.relu(x)
        # x = self.gcn2(x, edge_index)
        # x = torch.nn.functional.relu(x)

        if ptr is not None:
            idx_in_batch_1 = ptr[:-1] + node1
            idx_in_batch_2 = ptr[:-1] + node2
        else:
            idx_in_batch_1 = node1
            idx_in_batch_2 = node2

        x1 = torch.index_select(x, dim=0, index=idx_in_batch_1)
        x2 = torch.index_select(x, dim=0, index=idx_in_batch_2)
        emb_concat = torch.cat((x1, x2), dim=1)

        x = self.lin2(emb_concat)
        # x = torch.nn.functional.relu(x)
        # x = self.lin2(x)
        return x

def train_or_load_link_pred(loader):
    device = get_device()
    save_dst = './checkpoints/ppi/link_pred.model'
    model = LinkPredictorModel().to(device)
    loss_func = torch.nn.CrossEntropyLoss()

    if os.path.isfile(save_dst):
        model.load_state_dict(torch.load(save_dst))
    else:
        lr = 5.e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)
        num_epochs = 1000
        output_freq = 20
        train_model(model, False, optimizer, loader, loader, num_epochs, loss_func, save_dst, output_freq,
                    task=Task.LINK_PREDICTION)
    return model


def debug(model, graph, explain_edges):
    edge = explain_edges[:, 0].detach().cpu().tolist()
    print(f'explaining edge {edge}')
    subgraphx = SubgraphX(model, num_layers=1, exp_weight=5, m=20, t=50, task=Task.LINK_PREDICTION)
    explanation_set, mcts = subgraphx(graph, n_min=5, nodes_to_keep=[edge[0], edge[1]])
    print(f'explanation: {explanation_set}')
    sparsity_score = sparsity(graph, explanation_set)
    print(f'sparsity: {sparsity_score}')

    fidelity_score = fidelity(graph, explanation_set, model, task=Task.LINK_PREDICTION, nodes_to_keep=edge)
    print(f'fidelity: {fidelity_score}')


def collect_subgraphx_expl(model, graph, test_edge_list):
    device = get_device()

    path = './result_data/ppi/subgraphx_dict'
    if os.path.isfile(path):
        res_dict = load_data(path)
        return res_dict
    else:
        res_dict = dict()
        for edge in test_edge_list:
            edge = tuple(edge)
            res_dict[edge] = []
        # save_data(path, res_dict)

    # collect explanations for all nodes with a fixed n_min
    subgraphx = SubgraphX(model, num_layers=1, exp_weight=5, m=5, t=12, task=Task.LINK_PREDICTION)

    for n_min in [4, 6, 8, 10, 12]:
        counter = 1
        print(f'\nstarting {n_min}')
        for edge in test_edge_list:
            edge_tuple = tuple(edge)
            start_time = time.time()
            explanation_set, _ = subgraphx(graph, n_min=n_min, nodes_to_keep=edge)

            end_time = time.time()
            duration = end_time - start_time

            sparsity_score = sparsity(graph, explanation_set)
            fidelity_score = fidelity(graph, explanation_set, model, task=Task.LINK_PREDICTION,
                                      nodes_to_keep=edge)

            result_tuple = (explanation_set, sparsity_score, fidelity_score, duration)
            res_dict[edge_tuple] = res_dict[edge_tuple] + [result_tuple]
            print(f'finished node {counter} of {len(test_edge_list)}')
            counter += 1

    # save_data(path, res_dict)
    return res_dict



def main():
    seed = 0
    set_seed(seed)
    device = get_device()

    # get dataset
    edge_lists, y_lists, graph_train = prepare_dataset(seed)
    edge_list_train, edge_list_val, edge_list_test = edge_lists[0], edge_lists[1], edge_lists[2]
    y_train, y_val, y_test = y_lists[0], y_lists[1], y_lists[2]

    # get embedding
    emb_model = train_or_load_embedding(graph_train.edge_index)

    # make new graph containing embedding
    new_x = update_graph_features(emb_model)
    graph = Data(x=new_x, edge_index=graph_train.edge_index, edge_list_train=edge_list_train,
                 edge_list_val=edge_list_val, edge_list_test=edge_list_test, y_train=y_train, y_val=y_val,
                 y_test=y_test)
    loader = DataLoader([graph], batch_size=1, shuffle=False)

    # train model
    predictor_model = train_or_load_link_pred(loader)

    # performance on test set
    loss_func = torch.nn.CrossEntropyLoss()
    loss, acc = test(predictor_model, False, loader, loss_func, task=Task.LINK_PREDICTION)
    print(f'Test loss: {loss}, test acc: {acc}')

    # add softmax to model
    model = Sequential(
        'x, edge_index, x1, x2, ptr', [
            (predictor_model, 'x, edge_index, x1, x2, ptr -> x'),
            (Softmax(dim=1), 'x -> x')
        ]
    )

    # choose random instances to explain
    num_explanations = 5
    expl_idx = random.sample(list(np.arange(edge_list_test.shape[1])), k=num_explanations)
    explain_edges = torch.index_select(edge_list_test, dim=1, index=torch.tensor(expl_idx))
    explain_y = torch.index_select(y_test, dim=0, index=torch.tensor(expl_idx))

    # debug
    # debug(model, graph, explain_edges)

    # gather data
    # collect_subgraphx_expl(model, graph, explain_edges.T.detach().cpu().tolist())

    # load dict
    sx_dict = load_data('./result_data/ppi/subgraphx_dict')
    sx_sparsity, sx_fidelity = aggregate_fidelity_sparsity(sx_dict)
    sx_runtime = compute_avg_runtime(sx_dict)

    sparsity_list = [[4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12]]
    fidelity_list = [sx_fidelity]
    labels = ['SubgraphX']
    plot_results(sparsity_list, fidelity_list, labels, save_dst='./img/ppi/ppi_result2.png')

    data_str = f'Subgraphx runtime avg: {sx_runtime}'
    save_str(path='./result_data/ppi/runtime.txt', data=data_str)
    print(data_str)



if __name__ == '__main__':
    main()
