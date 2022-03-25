import os

import numpy as np
import torch
from torch import optim

from benchmarking import writeMatrix
from colab.geditt import gedit_main1, run_gedit_pre1
from colab.utils_functions import generate_dists, run_nmf_on_data_data, train_supervised_one_sample
from gedit_preprocess.MatrixTools import RescaleRows, Rescale_ZeroToOne
from layers.super_net import SuperNet

output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/" \
                "25.2_model/nmf_train_on_generated_data_W0"

output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/" \
                "25.2_model/2-reformated/tmp-eran/new_new3_mp"
ref_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/ref_mat_2/"
mixes_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/Nmf-Objects-2/"
true_prop_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/TrueProportions/"


def RescaleRowsOneMatrix(Ref, power):
    Combined = [Ref[z] for z in range(len(Ref))]
    Scaled = Rescale_ZeroToOne(Combined, power)
    ScaledRef = []
    for m in range(len(Combined)):
        ScaledRef.append(Scaled[m][:len(Ref[0])])
    return ScaledRef


def main_train_on_generated_data():
    lr = 0.001
    num_layers = 10
    n_iter = 600
    l1_value = True
    print("Start")
    refs = os.listdir(ref_folder)
    for ref_name in refs:

        ref_name = ref_name.replace(".tsv", "")
        ref_path = f"{ref_folder}/{ref_name}.tsv"
        tmp_file = f"{output_folder}/tmp.tsv"
        mixes_names = [h for h in os.listdir(mixes_folder)]
        for mix_name in mixes_names:
            print(f"on ref name: {ref_name} and mix: {mix_name}")
            mix_signature_folder = f"{output_folder}/{mix_name}"
            mix_true_prop_path = f"{true_prop_folder}/TrueProps{mix_name}"

            if not os.path.isdir(mix_signature_folder):
                os.mkdir(mix_signature_folder)
            mix_path = f"{mixes_folder}/{mix_name}"

            #ref_object, mix_object = gedit_main1(ref_path, mix_path)
            ref_object, mix_object = run_gedit_pre1(mix_path, ref_path)

            ref_data = np.asanyarray(ref_object)[1:, 1:].astype(float)
            mix_data = np.asanyarray(mix_object)
            train_data = [generate_dists(ref_data.T, file_index * 0.1) for file_index in range(25)]  # t, dist

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

            for train_index in range(len(train_data)):
                v_train_i = torch.from_numpy(train_data[train_index][0]).float()

                v_train_i = v_train_i.T
                resultV = np.zeros((v_train_i.shape[0] + 1, v_train_i.shape[1] + 1), dtype='U28')
                resultV[0, 1:] = mix_object[0][1:]
                resultV[1:, 1:] = v_train_i.detach().numpy()

                resultV[:, 0] = np.asanyarray(mix_object)[:, 0]
                writeMatrix(resultV, tmp_file)
                dist_train_i = torch.from_numpy(train_data[train_index][1]).float()

                _, Scaledv_train_i = run_gedit_pre1(tmp_file, ref_path, True)
                #_, Scaledv_train_i = run_gedit_pre1(v_train_i.detach().numpy().T.tolist(), ref_path, True)
                # _, Scaledv_train_i = RescaleRows(SigMix,
                #                 v_train_i.detach().numpy().T.tolist(), 0.0, False)
                Scaledv_train_i = np.asanyarray(Scaledv_train_i).T[1:, 1:]
                Scaledv_train_i = torch.from_numpy(Scaledv_train_i.astype(float)).float()

                loss_values = train_supervised_one_sample(Scaledv_train_i, dist_train_i, n_iter,
                                                          deep_nmf, optimizerADAM, False)
                if train_index % 24 == 0:
                    print(f"Start train index: {train_index} with loss: {loss_values[-1]}")

            output_path = f"{mix_signature_folder}/dnmf$train$GeneratedW0_{mix_name}_{ref_name}.tsv"
            run_nmf_on_data_data(mix_data[1:, 1:].astype(float).T, np.asanyarray(ref_object),
                                 output_path, deep_nmf, reformat_path=mix_true_prop_path)


if __name__ == '__main__':
    main_train_on_generated_data()
