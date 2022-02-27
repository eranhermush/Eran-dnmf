import os

import numpy as np
from os import listdir, mkdir
from os.path import isfile, join

SIG_LM_PATH = "/Users/Eran/Documents/generate_predictions/LM22.npy"


def generate_dists(file_index, signature_data, dist_folder, bulk_folder, std):
    # sig = np.load(sig_pickle)
    dist = np.asanyarray([np.random.dirichlet([1 for i in range(signature_data.shape[0])]) for i in range(100)],
                         dtype=np.float)

    t = dist.dot(signature_data)
    t += np.random.normal(0, std, t.shape)
    t = np.maximum(t, 0)
    np.save(join(dist_folder, "dist_" + str(file_index)), dist)
    np.save(join(bulk_folder, "t_" + str(file_index)), t)


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


def generate_signatures(sig_folder, output_folder):
    ref_files = [f for f in listdir(sig_folder) if isfile(join(sig_folder, f))]
    for signature in ref_files:
        sig = read_dataset(join(sig_folder, signature))
        np.save(join(output_folder, signature), sig)


def from_mix_input_to_signature_input(signature_path, dataset_path, output_path):
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
    np.save(output_path, (result.T))


def generate_signature_mixes(signature_folder, mix_folder, output_folder):
    signature_files = [f for f in listdir(signature_folder) if isfile(join(signature_folder, f))]
    mix_files = [f for f in listdir(mix_folder) if isfile(join(mix_folder, f))]

    for signature in signature_files:
        print("Working on " + signature)
        signature_name = signature.split(".")[0]
        output_folder_signature = join(output_folder, signature_name)
        mkdir(output_folder_signature)
        signature_path = join(signature_folder, signature)
        for mix_file in mix_files:
            mix_path = join(mix_folder, mix_file)
            output_path = join(output_folder_signature, mix_file)
            from_mix_input_to_signature_input(signature_path, mix_path, output_path)


def generate_resoucres(signature_data, sig_name, output_folder):
    dist_folder = join(output_folder, "resources_dist_" + sig_name)
    bulk_folder = join(output_folder, "resources_bulk_" + sig_name)
    if not (os.path.isdir(dist_folder) and os.path.isdir(bulk_folder)):
        mkdir(dist_folder)
        mkdir(bulk_folder)
        for i in range(50):
            generate_dists(i, signature_data, dist_folder, bulk_folder, std=i / 4)


def generate_resources_main(ref_dir, output_folder, signature_name=None):
    # ref_dir = "/Users/Eran/Documents/generate_predictions/lm22/"
    # output_folder = "/Users/Eran/Documents/generate_predictions/ref_gedit/resources_10/"

    ref_files = [f for f in listdir(ref_dir) if isfile(join(ref_dir, f))]
    if signature_name:
        ref_files = [f for f in ref_files if signature_name in f]
    for signature in ref_files:
        print(signature)
        signature_data = read_dataset(join(ref_dir, signature))[1:, 1:]
        signature_data = signature_data.T.astype(np.float)

        generate_resoucres(signature_data, signature, output_folder)



if __name__ == '__main__':
    # from_mix_input_to_signature_input("LM22.txt","CellMixtures.tsv", "CeelMix_lm22T")
    # main()

    # generate_signature_mixes("/Users/Eran/Documents/generate_predictions/lm22/",
    #                         "/Users/Eran/Documents/generate_predictions/Mixes/",
    #                        "/Users/Eran/Documents/generate_predictions/Signature_mixes/")

    # generate_signatures("/Users/Eran/Documents/generate_predictions/ref_gedit/gedit_signatures_nmf/",
    #                    "/Users/Eran/Documents/generate_predictions/RefNp/")
    generate_signatures("/Users/Eran/Documents/generate_predictions/ref_gedit/gedit_signatures_nmf/",
                        "/Users/Eran/Documents/generate_predictions/RefNp/")
