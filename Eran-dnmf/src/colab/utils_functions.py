import math

import torch

import numpy as np
from torch import nn

from benchmarking import writeMatrix
from colab.converter import main_format


def run_nmf_on_data_data(data_file, signature, output_path, deep_nmf, reformat_path=None):
    signature_data = signature[1:, 1:].astype(float).T
    samples, genes_count = data_file.shape
    cells_count, genes_count = signature_data.shape

    H_init_np = np.ones((samples, cells_count))

    H_init = torch.from_numpy(H_init_np).float()
    data_torch = torch.from_numpy(data_file).float()
    out = deep_nmf(H_init, data_torch)
    normalize_out = out / out.sum(axis=1)[:, None]

    result = np.zeros((out.shape[0] + 1, out.shape[1] + 1), dtype='U28')
    result[0, 1:] = signature[0][1:]
    result[1:, 1:] = normalize_out.detach().numpy()

    result[:, 0] = [f"Mixture_{i}" for i in range(len(result[:, 0]))]
    use_t = True
    if reformat_path is not None:
        ref_object_formated = main_format(result, reformat_path)
        #ref_object_formated[0] = "\tmix" + ref_object_formated[0]
        result = ref_object_formated
        use_t = False

    writeMatrix(result, output_path, use_t)


def create_mix_train(mix_data, train_data):
    genes_names_in_mix = [g[0] for g in mix_data[1:]]

    result = np.zeros((mix_data.shape[0] - 1, mix_data.shape[1] - 1), dtype=float)
    for gene_index in range(1, len(train_data)):
        gene_name = train_data[gene_index][0]
        if gene_name in genes_names_in_mix:
            sig_index = genes_names_in_mix.index(gene_name)
            result[sig_index] = train_data[gene_index][1:]

    return result


def read_dataset_data(file_data):
    data_file = []
    for line in file_data:
        splitLine = line.strip().split("\t")
        if len(splitLine) == 1:
            splitLine = line.strip().split(",")
        data_file.append(splitLine)
    data_file2 = np.array(data_file)
    return data_file2


def generate_dists(signature_data, std):
    dist = np.asanyarray([np.random.dirichlet([1 for i in range(signature_data.shape[0])]) for i in range(100)],
                         dtype=float)

    t = dist.dot(signature_data)
    t += np.random.normal(0, std, t.shape)
    t = np.maximum(t, 0)
    return t, dist


def train_supervised_one_sample(v_train, h_train, network_train_iterations, deep_nmf,
                                optimizer, verbose=False, print_every=100):
    n_h_rows, n_components = h_train.shape
    H_init_np = np.ones((n_h_rows, n_components))
    H_init = torch.from_numpy(H_init_np).float()

    criterion = nn.MSELoss(reduction="mean")
    # Train the Network
    loss_values = []
    for i in range(network_train_iterations):
        out = deep_nmf(H_init, v_train)
        loss = torch.sqrt(criterion(out, h_train))  # loss between predicted and truth

        if verbose:
            if (i % print_every == 0):
                print(i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)
        loss_values.append(loss.item())

    return loss_values


def train_supervised_one_sample_reformat(v_train, h_train, n_components_ref, cells_in_ref, network_train_iterations,
                                         deep_nmf,
                                         optimizer, true_prop_path, verbose=False, print_every=100):
    n_h_rows, n_components = h_train.shape
    H_init_np = np.ones((n_h_rows, n_components_ref))
    H_init = torch.from_numpy(H_init_np).float()

    criterion = nn.MSELoss(reduction="mean")

    # Train the Network
    loss_values = []
    for i in range(network_train_iterations):
        out = deep_nmf(H_init, v_train)

        """
        out_result = np.zeros((out.shape[0] + 1, out.shape[1] + 1), dtype='U28')
        out_result[0, 1:] = cells_in_ref
        out_result[1:, 1:] = out.detach().numpy()
        out_result[1:, 0] = [f"Mixture_{i}" for i in range(len(out[:, 0]))]

        reformated_out = main_format(out_result, true_prop_path)
        reformated_out[0] = "\tmix" + reformated_out[0]

        reformated_out = torch.from_numpy(read_dataset_data(reformated_out)[1:, 1:].astype(float)).float()
        loss = torch.sqrt(criterion(reformated_out, h_train))  # loss between predicted and truth
        """
        loss = torch.sqrt(criterion(out, h_train))  # loss between predicted and truth
        if verbose:
            if (i % print_every == 0):
                print(i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)
        loss_values.append(loss.item())

    return loss_values
