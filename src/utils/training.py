# Generic Code for training and Evaluating Neural Networks
from typing import Callable, Tuple, Union, List

import torch
from torch import optim
from torch_geometric.loader import DataLoader

from src.utils.task_enum import Task, Stage
from src.utils.utils import get_device


def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    correct = (pred == target)
    num_correct = correct.sum().item()
    num_items = correct.shape[0]
    return num_correct, num_items


def _episode_helper(model: torch.nn.Module, single_output: bool, optimizer: Union[optim.Optimizer, None],
                    loader: DataLoader, loss_func: Callable, stage: Stage = Stage.TRAINING,
                    task: Task = Task.GRAPH_CLASSIFICATION) -> Tuple[float, float]:
    device = get_device()
    model = model.to(device)

    loss = 0
    correct = 0
    num_items = 0

    for graph in iter(loader):
        if stage == Stage.TRAINING:
            optimizer.zero_grad()

        mask = None  # mask for node classification
        if task == Task.NODE_CLASSIFICATION:
            if stage == Stage.TRAINING:
                mask = graph.train_mask
            elif stage == Stage.VALIDATION:
                mask = graph.val_mask
            else:  # testing
                mask = graph.test_mask

        edge_list = None  # feature and targets for link prediction
        y_list = None
        if task == Task.LINK_PREDICTION:
            if stage == Stage.TRAINING:
                edge_list = graph.edge_list_train
                y_list = graph.y_train
            elif stage == Stage.VALIDATION:
                edge_list = graph.edge_list_val
                y_list = graph.y_val
            else:  # testing
                edge_list = graph.edge_list_test
                y_list = graph.y_test

        # forward pass
        if task == Task.NODE_CLASSIFICATION or task == Task.GRAPH_CLASSIFICATION:
            scores = model(graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device))
        else:  # link prediction
            scores = model(graph.x.to(device), graph.edge_index.to(device), edge_list[0].to(device),
                           edge_list[1].to(device))

        if single_output:
            scores = scores.squeeze()

        if task == Task.GRAPH_CLASSIFICATION:
            cur_loss = loss_func(scores, graph.y.to(device))
        elif task == Task.NODE_CLASSIFICATION:
            cur_loss = loss_func(scores[mask], graph.y.to(device)[mask])
        else:
            cur_loss = loss_func(scores, y_list.to(device))

        if stage == Stage.TRAINING:  # backward pass
            cur_loss.backward()
            optimizer.step()

        if single_output:  # retrieve predicted class with maximum score
            pred = torch.round(scores).detach().cpu()
        else:
            pred = torch.argmax(scores, dim=1).detach().cpu()

        if task == Task.GRAPH_CLASSIFICATION:  # compute accuracy of prediction for verbose output
            cur_correct, cur_num_items = compute_accuracy(pred, graph.y)
        elif task == Task.NODE_CLASSIFICATION:
            cur_correct, cur_num_items = compute_accuracy(pred[mask], graph.y[mask])
        else:
            cur_correct, cur_num_items = compute_accuracy(pred, y_list)

        correct += cur_correct
        num_items += cur_num_items
        loss += cur_loss.detach().cpu().item()

    loss /= len(loader)
    acc = correct / num_items

    return loss, acc


def _episode(model: torch.nn.Module, single_output: bool, optimizer: Union[optim.Optimizer, None], loader: DataLoader,
             loss_func: Callable, stage: Stage = Stage.TRAINING, task: Task = Task.GRAPH_CLASSIFICATION) \
             -> Tuple[float, float]:
    if stage == Stage.TRAINING:
        model.train()
    else:
        model.eval()

    if not stage == Stage.TRAINING:
        with torch.no_grad():
            return _episode_helper(model, single_output, optimizer, loader, loss_func, stage, task=task)
    else:
        return _episode_helper(model, single_output, optimizer, loader, loss_func, stage, task=task)


def test(model: torch.nn.Module, single_output: bool, test_loader: DataLoader, loss_func: Callable,
         task: Task = Task.GRAPH_CLASSIFICATION) -> Tuple[float, float]:
    loss, acc = _episode(model, single_output, None, test_loader, loss_func, stage=Stage.TESTING, task=task)
    return loss, acc


def train_model(model: torch.nn.Module, single_output: bool, optimizer: optim.Optimizer, train_loader: DataLoader,
                val_loader: DataLoader, num_epochs: int, loss_func: Callable, save_dst: str, output_freq: int = 1,
                task: Task = Task.GRAPH_CLASSIFICATION) \
                -> Tuple[List[float], List[float], List[float], List[float]]:
    best_epoch = -1
    best_val_acc = 0

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    # initial acc
    init_loss, init_acc = _episode(model, single_output, None, val_loader, loss_func, stage=Stage.VALIDATION, task=task)
    print(f'Initial Val Accuracy: {init_acc:.4f}, initial Loss: {init_loss}')

    for i in range(1, num_epochs + 1):

        # training
        cur_train_loss, cur_train_acc = _episode(model, single_output, optimizer, train_loader, loss_func,
                                                 stage=Stage.TRAINING, task=task)
        train_loss.append(cur_train_loss)
        train_acc.append(cur_train_acc)

        # eval
        cur_val_loss, cur_val_acc = _episode(model, single_output, None, val_loader, loss_func, stage=Stage.VALIDATION,
                                             task=task)
        val_loss.append(cur_val_loss)
        val_acc.append(cur_val_acc)
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            best_epoch = i
            torch.save(model.state_dict(), save_dst)  # save the best model

        # output
        if (i % output_freq) == 0:
            print(f'Epoch: {i}, \ttrain_loss: {train_loss[-1]:.4f} \tacc_train: {train_acc[-1]:.4f}'
                  + f' \tval_loss: {val_loss[-1]:.4f} \tval_acc: {val_acc[-1]:.4f}')

    print(f'Finished Training. Best Val Acc in epoch {best_epoch}: {best_val_acc:.4f}, Loss: '
          + f'{val_loss[best_epoch - 1]}')
    model.load_state_dict(torch.load(save_dst))  # load the best model

    return train_loss, train_acc, val_loss, val_acc


def train_emb(model, optimizer, loader, num_epochs, output_freq: int = 1):
    device = get_device()
    for i in range(1, num_epochs + 1):
        running_loss = 0
        model.train()
        for pos_rw, neg_rw in loader:
            pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # output
        if (i % output_freq) == 0:
            running_loss /= len(loader)
            print(f'Epoch: {i} \tloss: {running_loss}')

