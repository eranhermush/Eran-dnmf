from os import listdir
from os.path import isfile, join

import torch

import numpy as np


def writeMatrix(Matrix, File, use_t=True):
    fstream = open(File, "w+")
    for line in Matrix:
        if use_t:
            try:
                line = "\t".join([line[0]] + [str(round(float(m), 6)) for m in line[1:]])
            except:
                line = "\t".join([str(m) for m in line])

        fstream.write(line)
        fstream.write("\n")
    fstream.close()
    return


def read_dataset(file_path):
    data_file = []
    with open(file_path, "r") as f:
        for line in f:
            splitLine = line.strip().split("\t")
            if len(splitLine) == 1:
                splitLine = line.strip().split(",")
            data_file.append(splitLine)
    data_file = np.asanyarray(data_file)
    return data_file


def from_mix_input_to_signature_input(signature_path, dataset_path):
    signature_data = read_dataset(signature_path)
    signature_data = signature_data[1:]
    genes_in_sig = list(signature_data[:, 0])
    signature_data = signature_data[:, 1:]

    dataset_data = read_dataset(dataset_path)
    dataset_data = dataset_data[1:]
    genes_in_data = dataset_data[:, 0]
    dataset_data = dataset_data[:, 1:]
    dataset_data = dataset_data.astype(np.float)

    dataset_data = dataset_data.astype(np.float)

    result = np.zeros((signature_data.shape[0], dataset_data.shape[1]), dtype=np.float)
    for gene_index in range(len(genes_in_data)):
        gene_name = genes_in_data[gene_index]
        if gene_name in genes_in_sig:
            sig_index = genes_in_sig.index(gene_name)
            result[sig_index] = dataset_data[gene_index]
    return result.T


def run_nmf_on_data(data_path, signature_path, output_path, deep_nmf):
    signature = read_dataset(signature_path)
    # import pdb;pdb.set_trace()
    # data_file = read_dataset(data_path)[1:,1:]
    # data_file = data_file.astype(np.float).T
    data_file = from_mix_input_to_signature_input(signature_path, data_path)
    signature_data = signature[1:, 1:].T
    samples, genes_count = data_file.shape
    cells_count, genes_count = signature_data.shape

    H_init_np = np.ones((samples, cells_count))

    H_init = torch.from_numpy(H_init_np).float()
    data_torch = torch.from_numpy(data_file).float()
    out = deep_nmf(H_init, data_torch)
    normalize_out = out / out.sum(axis=1)[:, None]

    result = np.zeros((out.shape[0] + 1, out.shape[1]), dtype='U28')
    result[0,] = signature[0][1:]
    result[1:, ] = normalize_out.detach().numpy()
    writeMatrix(result, output_path)


def run_nmf_all(data_folder, signature_path, output_folder, sig_name):
    data_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
    for data_file in data_files:
        data_file_name = data_file.replace(".npy", "")
        data_name = data_file_name.replace(".tsv", "")
        print(data_name)
        # lm_CellMixtures.tsv_EPIC-BCIC.tsv.tsv
        output_file_name = f"dnmf_{data_name}.tsv_{sig_name}.tsv.tsv"
        run_nmf_on_data(join(data_folder, data_file), signature_path, join(output_folder, output_file_name))
