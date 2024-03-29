import os

import numpy as np
import torch
from torch import optim, nn

from benchmarking import read_dataset, writeMatrix
from colab.converter import main_format
from colab.geditt import gedit_main1, run_gedit_pre1
from colab.utils_functions import (
    generate_dists,
    run_nmf_on_data_data,
    train_supervised_one_sample,
    read_dataset_data,
    tensoring,
)
from gedit_preprocess.MatrixTools import readMatrix, getSharedRows
from layers.super_net import SuperNet

# output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/" \
#                "25.2_model/2-reformated/cross_val_on_other"
output_folder = (
    "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/"
    "25.2_model/2-reformated/supervised_with_new_gedit"
)
ref_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/ref_mat_2/"
mixes_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/Nmf-Objects-2/"
mixes_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/m/"
true_prop_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/TrueProportions/"


def get_matrices(mix_path, ref_path, use_gedit, use_all_genes=False):
    if use_gedit == "new_gedit":
        ref_object, mix_object = run_gedit_pre1(mix_path, ref_path, True)
    elif use_gedit:
        ref_object, mix_object = run_gedit_pre1(mix_path, ref_path, use_all_genes)
    else:
        ref_object_matrix = readMatrix(ref_path)
        mix_object_matrix = readMatrix(mix_path)
        mix_object, ref_object = getSharedRows(mix_object_matrix, ref_object_matrix)
        ref_object = np.asanyarray([ref_object_matrix[0]] + ref_object)
        mix_object = np.asanyarray([mix_object_matrix[0]] + mix_object)

    return ref_object, mix_object


def update_ref_on_genes(mixes_folder, ref_path):
    ref_object_matrix = readMatrix(ref_path)
    first_row = ref_object_matrix[0]
    mixes_names = [h for h in os.listdir(mixes_folder)]
    for mix_name in mixes_names:
        mix_path = f"{mixes_folder}/{mix_name}"
        mix_object_matrix = readMatrix(mix_path)
        _, ref_object_matrix = getSharedRows(mix_object_matrix, ref_object_matrix)
    return [first_row] + ref_object_matrix


def main_train_on_generated_data(use_gedit=True, useW0=True, output_prefix="", output_folder=output_folder):
    lr = 0.001
    num_layers = 10
    n_iter = 1000
    l1_value = True
    print("Start")
    refs = os.listdir(ref_folder)
    for ref_name in refs:

        ref_name = ref_name.replace(".tsv", "")
        ref_path = f"{ref_folder}/{ref_name}.tsv"
        tmp_file = f"{output_folder}/tmp.tsv"
        ref_object_matrix = update_ref_on_genes(mixes_folder, ref_path)
        ref_path = tmp_file
        writeMatrix(ref_object_matrix, tmp_file)

        mixes_names = [h for h in os.listdir(mixes_folder)]
        for mix_name in mixes_names:
            print(f"on ref name: {ref_name} and mix: {mix_name}")
            mix_signature_folder = f"{output_folder}/{mix_name}"
            mix_true_prop_path = f"{true_prop_folder}/TrueProps{mix_name}"

            if not os.path.isdir(mix_signature_folder):
                os.mkdir(mix_signature_folder)
            mix_path = f"{mixes_folder}/{mix_name}"

            ref_object, mix_object = get_matrices(mix_path, ref_path, use_gedit)

            ref_object_formated = main_format(ref_object, mix_true_prop_path)
            ref_object_formated[0] = "\tmix" + ref_object_formated[0]

            ref_object = read_dataset_data(ref_object_formated)[:, :-1]
            ref_data = ref_object[1:, 1:].astype(float)

            mix_data = np.asanyarray(mix_object)
            Y = read_dataset(mix_true_prop_path)
            kf = KFold(n_splits=10)
            loss_arr = []
            deep_nmf_arr = []
            for train_index, test_index in kf.split(mix_data[1:, 1:].T):
                X_train, X_test = (
                    mix_data[1:, 1:].astype(float).T[train_index],
                    mix_data[1:, 1:].astype(float).T[test_index],
                )
                y_train, y_test = Y[1:, 1:].astype(float)[train_index], Y[1:, 1:].astype(float)[test_index]

                features, n_components = ref_data.shape
                deep_nmf = SuperNet(num_layers, n_components, features, L1=l1_value, L2=True)
                deep_nmf_params = list(deep_nmf.parameters())
                for w in deep_nmf.parameters():
                    w.data.fill_(0.1)
                if useW0:
                    for w_index in range(len(deep_nmf_params)):
                        w = deep_nmf_params[w_index]
                        if w_index == 2:
                            w.data = torch.from_numpy(np.dot(ref_data.T, ref_data)).float()
                        elif w_index == 3:
                            w.data = torch.from_numpy(ref_data.T).float()
                        else:
                            w.data.fill_(1.0)
                optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)

                loss_values = train_supervised_one_sample(
                    torch.from_numpy(X_train).float(),
                    torch.from_numpy(y_train).float(),
                    n_iter,
                    deep_nmf,
                    optimizerADAM,
                    False,
                )

                output_path = f"{mix_signature_folder}/NoWnmf_cross_{mix_name}_{ref_name}.tsv"
                loss = run_nmf_on_data_data(
                    np.asanyarray(X_test),
                    np.asanyarray(ref_object),
                    output_path,
                    deep_nmf,
                    reformat_path=mix_true_prop_path,
                    y_value=torch.from_numpy(y_test).float(),
                )
                loss_arr.append(loss)
                deep_nmf_arr.append(deep_nmf)
            print(f"avg loss is {sum(loss_arr) / len(loss_arr)}")
            result_avg_loss = sum(loss_arr) / len(loss_arr)
            output_path_loss = f"{mix_signature_folder}/NoWDnmf$SupervisedAvg_{mix_name}_{ref_name}.tsv"
            if output_prefix:
                output_path_loss = f"{mix_signature_folder}/{output_prefix}_{mix_name}_{ref_name}.tsv"
            # writeMatrix(np.array([[result_avg_loss.tolist()]]), output_path_loss)
            find_result_with_ensemble(
                deep_nmf_arr,
                output_folder,
                mix_name,
                ref_name,
                mixes_folder,
                true_prop_folder,
                ref_path,
                use_gedit,
                ref_object.tolist(),
                output_prefix,
            )


def find_result_with_ensemble(
    nmf_array,
    output_folder,
    deep_mix_name,
    ref_name,
    mixes_folder,
    true_prop_folder,
    ref_path,
    use_gedit,
    ref_object_formated,
    output_prefix,
):
    mixes_names = [h for h in os.listdir(mixes_folder)]
    for mix_name in mixes_names:
        mix_true_prop_path = f"{true_prop_folder}/TrueProps{mix_name}"
        mix_signature_folder = f"{output_folder}/{mix_name}"
        if not os.path.isdir(mix_signature_folder):
            os.mkdir(mix_signature_folder)
        mix_path = f"{mixes_folder}/{mix_name}"
        ref_object, mix_object = get_matrices(mix_path, ref_path, use_gedit, True)
        if type(mix_object) == np.ndarray:
            mix_object = mix_object.tolist()
        mix_object, _ = getSharedRows(mix_object, ref_object_formated)

        mix_data = np.asanyarray(mix_object)
        Y = read_dataset(mix_true_prop_path)

        X = tensoring(mix_data[:, 1:].astype(float).T)
        Y = tensoring(Y[1:, 1:].astype(float))

        n_h_rows, n_components = Y.shape
        H_init_np = tensoring(np.ones((n_h_rows, n_components)))

        out_arr = [deep_nmf(H_init_np, X) for deep_nmf in nmf_array]
        normalize_out_arr = [out / out.sum(axis=1)[:, None] for out in out_arr]
        out = sum(normalize_out_arr) / len(normalize_out_arr)

        criterion = nn.MSELoss(reduction="mean")
        loss = torch.sqrt(criterion(out, Y))

        print(f"avg loss on {mix_name} is {loss}")
        output_path_loss = f"{mix_signature_folder}/GDnmf$SupervisedAvgFrom{deep_mix_name}_{mix_name}_{ref_name}.tsv"
        if output_prefix != "":
            output_path_loss = f"{mix_signature_folder}/{output_prefix}{deep_mix_name}_{mix_name}_{ref_name}.tsv"
        writeMatrix(np.array([[loss.tolist()]]), output_path_loss)

        output_path_h = f"{mix_signature_folder}/OUTGDnmf$SupervisedAvgFrom{deep_mix_name}_{mix_name}_{ref_name}.tsv"
        if output_prefix != "":
            output_path_h = f"{mix_signature_folder}/OUT${output_prefix}{deep_mix_name}_{mix_name}_{ref_name}.tsv"
        writeMatrix(out.detach().numpy(), output_path_h)


def generate_yes_no(bool_var):
    c = {True: "Yes", False: "No", "new_gedit": "NewG"}
    return c[bool_var]  # 'use' if bool_var else 'notUse'


if __name__ == "__main__":
    option_gedit = ["new_gedit", False, True]
    option_W0 = [False, True]
    for gedit in option_gedit:
        mix_signature_folder = f"{output_folder}/{gedit}"
        if not os.path.isdir(mix_signature_folder):
            os.mkdir(mix_signature_folder)

        for w0 in option_W0:
            output_string = f"Dnmf$SupervisedAvg${generate_yes_no(gedit)}Gedit${generate_yes_no(w0)}Wo$From"
            print(f"{generate_yes_no(gedit)}Gedit${generate_yes_no(w0)}")
            main_train_on_generated_data(
                use_gedit=gedit, useW0=w0, output_prefix=output_string, output_folder=mix_signature_folder
            )
