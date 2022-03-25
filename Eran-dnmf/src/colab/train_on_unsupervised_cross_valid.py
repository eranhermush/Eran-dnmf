import os
from itertools import product

import torch

import numpy as np
from torch import nn

from benchmarking import writeMatrix, read_dataset
from colab.converter import main_format
from colab.geditt import run_gedit_pre1
from colab.utils_functions import train_unsupervised, tensoring, read_dataset_data
from gedit_preprocess.MatrixTools import readMatrix, getSharedRows

output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/" \
                "25.2_model/2-reformated/unsupervised_w0_l1cross"
output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/" \
                "25.2_model/2-reformated/unsupervised_w0_l1cross1"
ref_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/ref_mat_2/"
mixes_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/Nmf-Objects-2/"
true_prop_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/TrueProportions/"


def main_train_on_generated_data(use_gedit=True, useW0=True, output_prefix=""):
    num_layers = 10
    n_iter = 500
    print("Start")
    refs = os.listdir(ref_folder)
    for ref_name in refs[1:]:

        ref_name = ref_name.replace(".tsv", "")
        ref_path = f"{ref_folder}/{ref_name}.tsv"
        mixes_names = [h for h in os.listdir(mixes_folder)]
        for mix_name in mixes_names:
            print(f"on ref name: {ref_name} and mix: {mix_name}")
            mix_signature_folder = f"{output_folder}/{mix_name}"
            mix_true_prop_path = f"{true_prop_folder}/TrueProps{mix_name}"

            if not os.path.isdir(mix_signature_folder):
                os.mkdir(mix_signature_folder)
            mix_path = f"{mixes_folder}/{mix_name}"

            if use_gedit:
                ref_object, mix_object = run_gedit_pre1(mix_path, ref_path)

            else:
                ref_object_matrix = readMatrix(ref_path)
                mix_object_matrix = readMatrix(mix_path)
                if useW0:
                    mix_object, ref_object = getSharedRows(mix_object_matrix, ref_object_matrix)
                    ref_object = np.asanyarray([ref_object_matrix[0]] + ref_object)
                    mix_object = np.asanyarray([mix_object_matrix[0]] + mix_object)
                else:
                    mix_object, ref_object = mix_object_matrix, ref_object_matrix

            ref_object_formated = main_format(ref_object, mix_true_prop_path)
            ref_object_formated[0] = "\tmix" + ref_object_formated[0]

            ref_object = read_dataset_data(ref_object_formated)[:, :-1]
            ref_data = torch.from_numpy(ref_object[1:, 1:].astype(float)).float()

            features, n_components = ref_data.shape
            mix_data = np.asanyarray(mix_object)[1:, 1:].astype(float)

            l_options = [0, 1, 2]
            for l1, l2 in product(l_options, l_options):
                output_path_h = f"{mix_signature_folder}/OUT${output_prefix}$L1{l1}$L2{l2}_{mix_name}_{ref_name}.tsv"
                if os.path.exists(output_path_h):
                    continue
                print("train")
                if useW0:
                    deep_nmf, dnmf_train_cost, dnmf_w, out_h = train_unsupervised(tensoring(mix_data).T, num_layers, n_iter,
                        n_components, ref_data=ref_data, l_1=l1, l_2=l2)
                else:
                    deep_nmf, dnmf_train_cost, dnmf_w, out_h = train_unsupervised(tensoring(mix_data).T, num_layers, n_iter,
                        n_components, l_1=l1, l_2=l2)
                criterion = nn.MSELoss(reduction="mean")

                dist_mix_i = torch.from_numpy(
                    read_dataset(mix_true_prop_path)[1:, 1:].astype(float)).float()

                loss = torch.sqrt(criterion(out_h, dist_mix_i))
                output_path_w = f"{mix_signature_folder}/Wdnf$Unsupervised$L1{l1}$L2{l2}_{mix_name}_{ref_name}.tsv"
                output_path_h = f"{mix_signature_folder}/Hdnf$Unsupervised$L1{l1}$L2{l2}_{mix_name}_{ref_name}.tsv"
                if output_prefix != "":
                    output_path_h = f"{mix_signature_folder}/OUT${output_prefix}$L1{l1}$L2{l2}_{mix_name}_{ref_name}.tsv"

                output_path_loss = f"{mix_signature_folder}/Dnmf$UnsupervisedNoGedit$L1{l1}$L2{l2}_{mix_name}_{ref_name}.tsv"
                if output_prefix != "":
                    output_path_loss = f"{mix_signature_folder}/{output_prefix}$L1{l1}$L2{l2}_{mix_name}_{ref_name}.tsv"

                print(f"l1: {l1}, l2: {l2} loss = {loss}")
                #writeMatrix(dnmf_w.detach().numpy(), output_path_w)
                writeMatrix(out_h.detach().numpy(), output_path_h)
                writeMatrix(np.array([[loss.tolist()]]), output_path_loss)


def generate_yes_no(bool_var):
    return 'Yes' if bool_var else 'No'


if __name__ == '__main__':
    option_gedit = [False, True]
    option_W0 = [False, True]

    for gedit in option_gedit:
        mix_signature_folder = f"{output_folder}/{gedit}"
        if not os.path.isdir(mix_signature_folder):
            os.mkdir(mix_signature_folder)
        for w0 in option_W0:
            output_string = f"Dnmf$Unupervised${generate_yes_no(gedit)}Gedit${generate_yes_no(w0)}Wo$"
            print(f"{generate_yes_no(gedit)}Gedit${generate_yes_no(w0)}W0")
            main_train_on_generated_data(use_gedit=gedit, useW0=w0, output_prefix=output_string)

