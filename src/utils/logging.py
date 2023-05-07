import pickle
from typing import Dict, Tuple, List

import numpy as np


def save_data(path: str, data: Dict):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def save_str(path: str, data: str):
    with open(path, 'w') as f:
        f.write(data)

def load_data(path: str) -> Dict:
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
        return loaded_dict

"""
For every node in the test set multiple data points are collected. Each point consists of: 
(1) explanation node set
(2) sparsity
(3) fidelity
(4) time to compute explanation (seconds)
"""
def aggregate_fidelity_sparsity(res_dict: Dict) -> Tuple[List[float], List[float]]:
    # initialize, assume all node list have same length and adhere to format
    sparsity = []
    fidelity = []
    node_list = next(iter(res_dict.values()))
    for _ in range(len(node_list)):
        sparsity.append(0)
        fidelity.append(0)

    # gather values
    for node, node_list in res_dict.items():
        for i, res_tuple in enumerate(node_list):
            explanation_set, sparsity_score, fidelity_score, duration = res_tuple
            sparsity[i] += sparsity_score
            fidelity[i] += fidelity_score

    # normalize
    num_nodes = len(res_dict)
    for i in range(len(sparsity)):
        sparsity[i] /= num_nodes
        fidelity[i] /= num_nodes

    # rearrange order such that sparsity is ascending
    idx = np.argsort(sparsity)
    sparsity = list(np.array(sparsity)[idx])
    fidelity = list(np.array(fidelity)[idx])

    return sparsity, fidelity


def compute_avg_runtime(res_dict: Dict) -> float:
    runtime = 0
    num_items = 0

    for node, node_list in res_dict.items():
        for i, res_tuple in enumerate(node_list):
            explanation_set, sparsity_score, fidelity_score, duration = res_tuple
            runtime += duration
            num_items += 1
    result = runtime / num_items
    return result
