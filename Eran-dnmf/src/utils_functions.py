import os

import numpy as np
import torch.optim as optim
import math
import pickle as pickle
from os import listdir
from os.path import isfile, join
import torch
import torch.nn as nn

from gedit_preprocess.GEDIT import gedit_main
from generate_preds import generate_resources_main
from layers.super_net import SuperNet


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
        loss = criterion(out, h_train)  # loss between predicted and truth

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


def from_file_to_torch(filename):
    np_object = np.load(filename)
    return torch.from_numpy(np_object).float()


def train_model(signature_name, resorces_folder, folder_to_save):
    v_trains = [from_file_to_torch(f"{resorces_folder}/resources_bulk_{signature_name}/t_{file_index}.npy") for
                file_index in range(50)]
    dist_trains = [from_file_to_torch(f"{resorces_folder}/resources_dist_{signature_name}/dist_{file_index}.npy") for
                   file_index in range(50)]
    n_iters = [500, 1000, 2000]
    num_layers_options = [5, 10, 20]
    l1 = [False, True]
    lr = 0.001
    save_folder = f"{folder_to_save}/{signature_name}"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    dist_train_tensor = dist_trains[0]
    v_train = v_trains[0]
    n_h_rows, n_components = dist_train_tensor.shape
    features = v_train.shape[1]
    print(f"Working on {signature_name}")
    for n_iter in n_iters:
        for num_layers in num_layers_options:
            for l1_value in l1:
                checkout_filename = f"{signature_name}_{n_iter}_{num_layers}_{l1_value}"

                deep_nmf = SuperNet(num_layers, n_components, features, L1=l1_value, L2=True)
                start_index = 0
                for w in deep_nmf.parameters():
                    w.data.fill_(1.0)
                optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)

                if (os.path.exists(f"{save_folder}/{checkout_filename}.pkl")):
                    model_dict = load_nmf(save_folder, checkout_filename)
                    if (len(v_trains) - 1 == model_dict["train_index"]):
                        print(
                            f"File exists: signature_name: {signature_name} n_iter: {n_iter}num_layers: {num_layers} l1_value: {l1_value}")
                        continue
                    else:
                        start_index = model_dict["train_index"]
                        optimizerADAM = model_dict["optimizer"]
                        deep_nmf = model_dict["model"]

                for train_index in range(start_index, len(v_trains)):
                    v_train_i = v_trains[train_index]
                    dist_train_i = dist_trains[train_index]
                    print(f"Start train index: {train_index} with num_layers: {num_layers} and n_iter = {n_iter}")
                    loss_values = train_supervised_one_sample(v_train_i, dist_train_i, n_iter, deep_nmf, optimizerADAM,
                                                              True, print_every=1001)
                    if (train_index % 5 == 0):
                        save_checkpoint(deep_nmf, optimizerADAM, checkout_filename, train_index, save_folder)
                save_checkpoint(deep_nmf, optimizerADAM, checkout_filename, train_index, save_folder)
                print(
                    f"Finish  signature_name: {signature_name} n_iter: {n_iter}num_layers: {num_layers} l1_value: {l1_value}")
    print(f"Finish working on {signature_name}")


def save_checkpoint(nmf_obj, optimizer, sig_name, train_index, folder):
    checkpoint = {
        "model": nmf_obj,
        "optimizer": optimizer,
        "sig_name": sig_name,
        "train_index": train_index
    }
    filename = f"{folder}/{sig_name}.pkl"
    print('Saving checkpoint to "%s"' % filename)
    with open(filename, "wb") as f:
        pickle.dump(checkpoint, f)


def load_nmf(folder, sig_name):
    filename = f"{folder}/{sig_name}.pkl"

    with open(filename, "rb") as f:
        model = pickle.load(f)

    return model


def main(signature_names):
    output_folder = '../resources/gedit/NMF-obj'
    signature_folder = '../resources/gedit/refMat'
    mixes_folder = '../resources/gedit/Mixes'
    mix_files_name = [f.split(".")[0] for f in listdir(mixes_folder) if isfile(join(mixes_folder, f))]
    resources_folder = "../resources/gedit/resources/"
    gedit_signature_folder = "../resources/gedit/resources/signatures"
    for signature_name in signature_names:
        gedit_main(signature_folder, mixes_folder, resources_folder, signature_name)
        for mix_name in mix_files_name:
            # CellMixtures.tsv_EPIC-BCIC.tsv_50_Entropy_0.0_ScaledMix.tsv
            signature_gedit_name = f"{mix_name}.tsv_{signature_name}.tsv_50_Entropy_0.0_ScaledRef.tsv"
            generate_resources_main(gedit_signature_folder, resources_folder, signature_gedit_name)
            train_model(signature_gedit_name, resources_folder[:-1], output_folder)


if __name__ == '__main__':
    # main(["LM22-Full"])
    # main(["HPCA-Stromal"])
    # main(["10XImmune"])
    main(["HPCA-Blood"])
