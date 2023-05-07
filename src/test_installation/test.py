from dig.xgraph.dataset import SynGraphDataset

import torch
from torch.nn import ReLU, Linear, Softmax
from torch_geometric.nn import Sequential, GCNConv

from dig.xgraph.method import SubgraphX
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method.subgraphx import PlotUtils
from dig.xgraph.method.subgraphx import find_closest_node_result

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main():
    # download dataset and test_installation
    dataset = SynGraphDataset('./datasets', 'BA_shapes')
    dataset.data.x = dataset.data.x.to(torch.float32)
    dataset.data.x = dataset.data.x[:, :1]
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    print(f'dim node {dim_node}, dim_edge {dim_edge}, num_classes {num_classes}')

    # model (untrained)
    model = Sequential(
        'x, edge_index', [
            (GCNConv(dim_node, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            Linear(64, num_classes),
            Softmax(dim=1),
        ]
    )
    model.to(device)

    # explainer
    explainer = SubgraphX(model, num_classes=num_classes, device=device, explain_graph=False, reward_method='nc_mc_l_shapley')

    # explanation
    x_collector = XCollector()
    index = -1
    node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
    data = dataset[0]
    print(data)

    # Visualization
    max_nodes = 5
    node_idx = node_indices[20]
    print(f'explain graph node {node_idx}')
    data.to(device)
    logits = model(data.x, data.edge_index)
    prediction = logits[node_idx].argmax(-1).item()
    print(f'prediction: {logits}, pred for node {prediction}')


    _, explanation_results, related_preds = explainer(data.x, data.edge_index, node_idx=node_idx, max_nodes=max_nodes)

    # explanation_results = explanation_results[prediction]
    # explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)
    #
    # plotutils = PlotUtils(dataset_name='ba_shapes', is_show=True)
    # explainer.visualization(explanation_results, max_nodes=max_nodes, plot_utils=plotutils, y=data.y)


if __name__ == '__main__':
    main()


