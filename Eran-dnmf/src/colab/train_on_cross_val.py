import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import optim

from benchmarking import read_dataset
from colab.converter import main_format
from colab.geditt import gedit_main1
from colab.utils_functions import generate_dists, run_nmf_on_data_data, train_supervised_one_sample, read_dataset_data
from layers.super_net import SuperNet

output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/" \
                "25.2_model/2-reformated/cross_val"
ref_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/ref_mat_2/"
mixes_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/Nmf-Objects-2/"
true_prop_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/TrueProportions/"


def main_train_on_generated_data(use_gedit=True):
    lr = 0.001
    num_layers = 10
    n_iter = 1000
    l1_value = True
    print("Start")
    refs = os.listdir(ref_folder)
    for ref_name in refs:

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

            ref_object, mix_object = gedit_main1(ref_path, mix_path)
            ref_object_formated = main_format(ref_object, mix_true_prop_path)
            ref_object_formated[0] = "\tmix" + ref_object_formated[0]

            ref_object = read_dataset_data(ref_object_formated)[:, :-1]
            ref_data = ref_object[1:, 1:].astype(float)


            mix_data = np.asanyarray(mix_object)
            Y = read_dataset(mix_true_prop_path)
            kf = KFold(n_splits=10)
            loss_arr = []
            for train_index, test_index in kf.split(mix_data[1:, 1:].T):
                X_train, X_test = mix_data[1:, 1:].astype(float).T[train_index], mix_data[1:, 1:].astype(float).T[
                    test_index]
                y_train, y_test = Y[1:, 1:].astype(float)[train_index], Y[1:, 1:].astype(float)[test_index]

                features, n_components = ref_data.shape
                deep_nmf = SuperNet(num_layers, n_components, features, L1=l1_value, L2=True)
                deep_nmf_params = list(deep_nmf.parameters())
                for w_index in range(len(deep_nmf_params)):
                    w = deep_nmf_params[w_index]
                    if w_index == 2:
                        w.data = torch.from_numpy(np.dot(ref_data.T, ref_data)).float()
                    elif w_index == 3:
                        w.data = torch.from_numpy(ref_data.T).float()
                    else:
                        w.data.fill_(1.0)
                optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)


                loss_values = train_supervised_one_sample(torch.from_numpy(X_train).float(),
                                torch.from_numpy(y_train).float(), n_iter, deep_nmf, optimizerADAM, False)


                output_path = f"{mix_signature_folder}/dnmf_cross$train$GeneratedW0_{mix_name}_{ref_name}.tsv"
                loss = run_nmf_on_data_data(np.asanyarray(X_test), np.asanyarray(ref_object),
                    output_path, deep_nmf, reformat_path=mix_true_prop_path, y_value=torch.from_numpy(y_test).float())
                loss_arr.append(loss)
            print(f"avg loss is {sum(loss_arr) / len(loss_arr)}")


if __name__ == '__main__':
    main_train_on_generated_data()
