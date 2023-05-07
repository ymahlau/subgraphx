import multiprocessing

from mutag import download_and_prepare_dataset, split_dataset, train_model_or_load, collect_subgraphx_expl, collect_gnn_expl
from src.utils.logging import load_data, save_data


def process_parallel():
    n = 5  # Increase with care
    workload = [(i, n) for i in range(n)]
    with multiprocessing.Pool(processes=n) as pool:
        pool.map(worker, workload)


def worker(workload):
    i, n = workload

    dataset = download_and_prepare_dataset()

    batch_size = 188
    train_loader, dev_loader, test_loader, dev_list = split_dataset(dataset, batch_size, (0.8, 1.0))

    num_graphs = len(dev_list)
    start_index = int(num_graphs * (i / n))
    end_index = int(num_graphs * ((i + 1) / n))
    print(f'worker {i} doing graphs in range({start_index},{end_index})')

    model_type = 'gin'

    model, loss_func = train_model_or_load(train_loader, dev_loader, model_type, add_softmax=True)

    # Stats for SubgraphX
    path_subgx = f'./result_data/mutag/{model_type}_subgxall_{i}'
    collect_subgraphx_expl(model, dev_list[start_index:end_index], path_subgx, workerno=i)
    print(f"Worker {i} finished")


def join_dicts(n: int, path: str):
    res_dict = {}
    counter = 0
    for i in range(n):
        cur_dict = load_data(path+'_'+str(i))
        for key, value in cur_dict.items():
            res_dict[counter] = value
            counter += 1
    print(len(res_dict))
    save_data(path, res_dict)




if __name__ == '__main__':
    #process_parallel()
    model_type = 'gin'
    join_dicts(n=5, path=f'./result_data/mutag/{model_type}_subgxall')
