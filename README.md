# SubgraphX Reproducibility

Reproducing results from the [SubgraphX](http://proceedings.mlr.press/v139/yuan21c.html) paper including a full re-implementation of the SubgraphX algorithm in PyTorch-Geometric.

The author's implementation can be found [here](https://github.com/divelab/DIG/blob/dig/dig/xgraph/method/subgraphx.py).

# Installation
For our experiments we use parts of the [DIG](https://github.com/divelab/DIG) library which restricts compatibility. Therefore, we recommend the following installation process:

```
conda env create -f environment.yml
conda activate subgraphx
```
To run the scripts from the project's root directory, make its path available to python:
```
export PYTHONPATH=.
```

