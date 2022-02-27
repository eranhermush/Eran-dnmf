import math
import os

import numpy as np
import torch
from torch import optim, nn

from benchmarking import read_dataset, writeMatrix
from colab.converter import main_format
from colab.geditt import gedit_main1
from layers.super_net import SuperNet


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


def train_supervised_one_sample_reformat(v_train, h_train, n_components_ref, cells_in_ref, network_train_iterations, deep_nmf,
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


def run_nmf_on_data_data(data_file, signature, output_path, deep_nmf):
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
    writeMatrix(result, output_path)


output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/25.2_model/nmf_train_on_real_data_W0"
ref_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/ref_mat_2/"
mixes_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/Nmf-Objects-2/"
true_prop_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/TrueProportions/"


def main():
    lr = 0.001
    num_layers = 10
    n_iter = 7000
    l1_value = True

    refs = os.listdir(ref_folder)
    for ref_name in refs:

        ref_name = ref_name.replace(".tsv", "")
        ref_path = f"{ref_folder}/{ref_name}.tsv"
        signature = read_dataset(ref_path)
        signature_data = signature[1:, 1:].T
        signature_data = signature_data.astype(float)

        mixes_names = [h for h in os.listdir(mixes_folder)]
        for mix_name in mixes_names:
            print(f"on ref name: {ref_name} and mix: {mix_name}")
            mix_signature_folder = f"{output_folder}/{mix_name}"
            mix_true_prop_path = f"{true_prop_folder}/TrueProps{mix_name}"
            if not os.path.isdir(mix_signature_folder):
                os.mkdir(mix_signature_folder)
            # dnmf$500$5$False_CellMixtures.tsv_10XImmune.tsv
            mix_path = f"{mixes_folder}/{mix_name}"

            ref_object, mix_object = gedit_main1(ref_path, mix_path)

            ref_object_formated = main_format(ref_object, mix_true_prop_path)
            ref_object_formated[0] = "\tmix" + ref_object_formated[0]

            ref_object = read_dataset_data(ref_object_formated)[:, :-1]
            ref_data = torch.from_numpy(ref_object[1:, 1:].astype(float)).float()
            # ref_data[:, :-1][5], ref_data[5]
            # ref_object = torch.from_numpy(read_dataset_data(ref_object_formated)[1:, 1:].astype(float)).float()

            genes_in_ref = ref_object[0][1:]
            genes_names = [g[0] for g in mix_object[1:]]
            # ref_data = np.asanyarray(ref_object)[1:, 1:].astype(float)

            mix_data = np.asanyarray(mix_object)

            mixes_names_copy = list(mixes_names)
            mixes_names_copy.remove(mix_name)
            # TruePropsPBMC1NormMix.tsv
            # PBMC2NormMix.tsv
            features, n_components = ref_data.shape

            deep_nmf = SuperNet(num_layers, n_components, features, L1=l1_value, L2=True)
            start_index = 0
            # list(list(deep_nmf.parameters())[3].detach().numpy())
            # np.dot(ref_data.T, ref_data).shape = 5*5
            deep_nmf_params = list(deep_nmf.parameters())
            for w_index in range(len(deep_nmf_params)):
                w = deep_nmf_params[w_index]
                if w_index == 2:
                    w.data = torch.from_numpy(np.dot(ref_data.T, ref_data))
                elif w_index == 3:
                    w.data = ref_data.T
                else:
                    w.data.fill_(1.0)

            optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)

            for train_index in range(len(mixes_names_copy)):
                train_mix_name = mixes_names_copy[train_index]
                train_mix_path = f"{mixes_folder}/{train_mix_name}"
                mix_train_data = read_dataset(train_mix_path)
                converted_mix_data = torch.from_numpy(create_mix_train(mix_data, mix_train_data)).float()

                dist_train_i_path = f"{true_prop_folder}/TrueProps{train_mix_name}"
                dist_train_i = read_dataset(dist_train_i_path)  # [1:, 1:].astype(float)
                data_formated = main_format(dist_train_i, mix_true_prop_path)
                data_formated[0] = "\tmix" + data_formated[0]

                dist_train_i_formated = torch.from_numpy(read_dataset_data(data_formated)[1:, 1:].astype(float)).float()[:, :-1]

                loss_values = train_supervised_one_sample_reformat(converted_mix_data.T, dist_train_i_formated,
                                                                   n_components, genes_in_ref, n_iter,
                                                                   deep_nmf, optimizerADAM, mix_true_prop_path, False,
                                                                   print_every=200)
                print(f"Start train index: {train_index} with loss: {loss_values[-1]}")

            output_path = f"{mix_signature_folder}/dnmf$train$datasetW0_{mix_name}_{ref_name}.tsv"
            run_nmf_on_data_data(mix_data[1:, 1:].astype(float).T, np.asanyarray(ref_object), output_path, deep_nmf)


if __name__ == '__main__':
    main()
