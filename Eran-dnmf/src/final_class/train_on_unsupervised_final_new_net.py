import os

import numpy as np
import torch
from torch import nn

from benchmarking import writeMatrix, read_dataset
from colab.converter import main_format
from colab.geditt import run_gedit_pre1
from colab.utils_functions import tensoring, read_dataset_data, train_unsupervised_new
from gedit_preprocess.MatrixTools import readMatrix, getSharedRows

ref_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/ref_mat_3/"
mixes_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/17.5/Nmf-Objects-2/"
true_prop_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/TrueProportions/"


def main_train_on_generated_data(output_folder, use_gedit=True, useW0=True, output_prefix="", w1_opt=None):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    num_layers = 4
    n_iter = 1000
    l1 = 0
    l2 = 1
    print("Start")

    refs = os.listdir(ref_folder)
    for ref_name in refs:

        ref_name = ref_name.replace(".tsv", "")
        ref_path = f"{ref_folder}/{ref_name}.tsv"
        mixes_names = [h for h in os.listdir(mixes_folder)]
        for mix_name in mixes_names:
            try:
                print(f"on ref name: {ref_name} and mix: {mix_name}")

                mix_signature_folder = f"{output_folder}/{mix_name}"
                mix_true_prop_path = f"{true_prop_folder}/TrueProps{mix_name}"
                output_path_h = f"{mix_signature_folder}/OUT${output_prefix}_{mix_name}_{ref_name}.tsv"
                output_path_loss = (
                    f"{mix_signature_folder}/Dnmf$UnsupervisedNoGedit$L1{l1}$L2{l2}_{mix_name}_{ref_name}.tsv"
                )
                if os.path.exists(output_path_h):
                    print("continue")
                #    continue
                if not os.path.isdir(mix_signature_folder):
                    os.mkdir(mix_signature_folder)

                mix_path = f"{mixes_folder}/{mix_name}"

                if use_gedit == True:
                    ref_object, mix_object = run_gedit_pre1(mix_path, ref_path)
                elif type(use_gedit) == int:
                    ref_object, mix_object = run_gedit_pre1(mix_path, ref_path, NumSigs=str(use_gedit))
                else:
                    ref_object_matrix = readMatrix(ref_path)
                    mix_object_matrix = readMatrix(mix_path)
                    if useW0:
                        mix_object, ref_object = getSharedRows(mix_object_matrix[1:], ref_object_matrix[1:])
                        ref_object = np.asanyarray([ref_object_matrix[0]] + ref_object)
                        mix_object = np.asanyarray([mix_object_matrix[0]] + mix_object)
                    else:
                        mix_object, ref_object = mix_object_matrix, ref_object_matrix

                ref_object_formated = main_format(ref_object, mix_true_prop_path)
                ref_object_formated[0] = "\tmix" + ref_object_formated[0]

                ref_object = read_dataset_data(ref_object_formated)[:, :-1]
                ref_data = torch.from_numpy(ref_object[1:, 1:].astype(float)).float()
                mix_data = np.asanyarray(mix_object)[1:, 1:].astype(float)

                features, n_components = ref_data.shape
                dist_mix_i = torch.from_numpy(read_dataset(mix_true_prop_path)[1:, 1:].astype(float)).float()
                writeMatrix(ref_data.detach().numpy(), output_path_h)

                deep_nmf, dnmf_train_cost, dnmf_w, out_h = train_unsupervised_new(
                    tensoring(mix_data).T,
                    num_layers,
                    n_iter,
                    n_components,
                    ref_path,
                    mix_object,
                    output_folder,
                    ref_data=ref_data,
                    l_1=l1,
                    l_2=l2,
                    w1_opt=w1_opt,
                    dist_mix_i=dist_mix_i,
                )
                criterion = nn.MSELoss(reduction="mean")

                loss = torch.sqrt(criterion(out_h, dist_mix_i))

                print(f"l1: {l1}, l2: {l2} loss = {loss}")
                writeMatrix(out_h.detach().numpy(), output_path_h)
                writeMatrix(np.array([[loss.tolist()]]), output_path_loss)
            except RuntimeError as e:
                print(f"there was an excpetion: {e}")


def generate_yes_no(bool_var):
    c = {True: "Yes", False: "No", "new_gedit": "NewG"}
    return c.get(bool_var, bool_var)


if __name__ == "__main__":
    output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/" "17.5/2_results"
    option_gedit = [True, False]  # , 500]
    option_W0 = [True]  # , False]
    options_w1 = ["last", "1", "algo", None]
    print("W1 algo")
    for gedit in option_gedit:
        mix_signature_folder = f"{output_folder}/{gedit}"
        for w0 in option_W0:
            for w1_opt in options_w1:
                output_string = f"Dnmf$Unupervised$W=W.{w1_opt}${generate_yes_no(gedit)}Gedit${generate_yes_no(w0)}Wo$"
                print(f"{generate_yes_no(gedit)}Gedit${generate_yes_no(w0)} w1 {w1_opt}")
                main_train_on_generated_data(
                    mix_signature_folder, use_gedit=gedit, useW0=w0, output_prefix=output_string, w1_opt=w1_opt
                )
