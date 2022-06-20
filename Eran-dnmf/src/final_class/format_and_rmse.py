import os

import torch

from torch import nn

from benchmarking import read_dataset, writeMatrix
from colab.converter import main_format
from colab.utils_functions import read_dataset_data
from gedit_preprocess.MatrixTools import readMatrix

mixes_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/25.3/r_results/500"
true_prop_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/TrueProportions"
output_folder = (
    "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/25.3/2_results_withR_new/500"
)


def format_and_rmse(mix_path, mix_true_prop_path, output_prefix, output_dir):
    mix_name, ref_name, _ = mix_path.split("/")[-1].split("_", 2)

    mix_object = readMatrix(mix_path)
    mix_object_formated = main_format(mix_object, mix_true_prop_path)
    mix_object_formated[0] = "\tmix" + mix_object_formated[0]

    mix_object = read_dataset_data(mix_object_formated)[:, :-1]
    mix_data = torch.from_numpy(mix_object[1:, 1:].astype(float)).float()

    Y = read_dataset(mix_true_prop_path)
    Y_data = torch.from_numpy(Y[1:, 1:].astype(float)).float()

    criterion = nn.MSELoss(reduction="mean")
    loss = torch.sqrt(criterion(mix_data, Y_data))

    output_path_loss = f"{output_dir}/{output_prefix}_{mix_name}_{ref_name}.tsv"
    writeMatrix([[loss.detach().numpy()]], output_path_loss)


def main(algo_mix_folder, out_folder, true_prop_folder):
    algos = os.listdir(algo_mix_folder)
    for algo in algos:
        mix_folder = f"{algo_mix_folder}/{algo}"
        mixes_names = [h for h in os.listdir(mix_folder)]
        for mix_name in mixes_names:
            short_mix_name = mix_name.split("_", 2)[0]
            mix_true_prop_path = f"{true_prop_folder}/TrueProps{short_mix_name}"
            if not mix_true_prop_path.endswith(".tsv"):
                mix_true_prop_path = mix_true_prop_path + ".tsv"
            mix_path = f"{mix_folder}/{mix_name}"
            output_prefix = str(algo)
            output_dir = f"{out_folder}/{short_mix_name}"
            format_and_rmse(mix_path, mix_true_prop_path, output_prefix, output_dir)


if __name__ == "__main__":
    main(mixes_folder, output_folder, true_prop_folder)
