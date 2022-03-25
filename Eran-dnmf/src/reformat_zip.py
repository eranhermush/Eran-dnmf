import csv
import shutil
from os.path import isdir

import numpy as np
import os

from generate_preds import read_dataset


def update_filename(filename, dataset):
    name, tsv = filename.split(".")
    signature, iterations, layers, l1 = name.split("_")
    new_name = "$".join(["dnmf", iterations, layers, l1])
    result = "_".join([new_name, dataset, signature])
    return result + "." + tsv


def filenmae_to_format(base_folder):
    signatures_folders = os.listdir(base_folder)
    for sign_folder in signatures_folders:
        cells = os.listdir(f"{base_folder}/{sign_folder}")
        for cell in cells:
            cell_folder = f"{base_folder}/{sign_folder}/{cell}"
            cell_files = os.listdir(cell_folder)
            for file_name in cell_files:
                new_name = update_filename(file_name, cell)
                os.rename(f"{cell_folder}/{file_name}", f"{cell_folder}/{new_name}")


def update_suffix(base_folder, old_suffix, new_suffix):
    signatures_folders = os.listdir(base_folder)
    for sign_folder in signatures_folders:
        cells = os.listdir(f"{base_folder}/{sign_folder}")
        for cell in cells:
            cell_folder = f"{base_folder}/{sign_folder}/{cell}"
            cell_files = os.listdir(cell_folder)
            for file_name in cell_files:
                new_name = file_name.replace(old_suffix, new_suffix)
                os.rename(f"{cell_folder}/{file_name}", f"{cell_folder}/{new_name}")


def writeMatrix(Matrix, File):
    fstream = open(File, "w+")
    for line in Matrix:
        try:
            line = "\t".join([line[0]] + [str(round(float(m), 6)) for m in line[1:]])
        except:
            line = "\t".join([str(m) for m in line])

        fstream.write(line)
        fstream.write("\n")
    fstream.close()
    return


def update_file_data(base_folder):
    signatures_folders = os.listdir(base_folder)
    for sign_folder in signatures_folders:
        cells = os.listdir(f"{base_folder}/{sign_folder}")
        for cell in cells:
            cell_folder = f"{base_folder}/{sign_folder}/{cell}"
            cell_files = os.listdir(cell_folder)
            for file_name in cell_files:
                data = read_dataset(f"{cell_folder}/{file_name}")
                new_data = np.zeros((data.shape[0], data.shape[1] + 1), dtype='U28')
                new_data[:, 1:] = data
                new_data[:, 0] = [f"Mixture_{i}" for i in range(len(new_data[:, 0]))]
                writeMatrix(new_data, f"{cell_folder}/{file_name}")


def main(base_folder):
    update_suffix(base_folder, ".pkl", ".tsv")
    filenmae_to_format(base_folder)
    update_file_data(base_folder)


def get_best_file_in_folder(folder, output_folder, function_name, to_max=False, is_gedit=None):
    best_name = ""
    best_value = 0
    start = True
    factor = -1
    if to_max:
        factor = 1
    files = os.listdir(folder)
    for file_name in files:
        if is_gedit is not None:
            gedit_in_filname = "gedit" in file_name.lower()
            if gedit_in_filname != is_gedit:
                continue
        file_function = file_name.split("_")[0]
        if file_function == function_name:
            file_path = f"{folder}/{file_name}"
            file_size = get_file_data(file_path)
            if file_size is None:
                continue
            file_size = file_size * factor
            if start:
                start = False
                best_value = file_size
                best_name = file_name
            elif file_size > best_value:
                best_value = file_size
                best_name = file_name
    if best_name != "":
        shutil.copyfile(f"{folder}/{best_name}", f"{output_folder}/{best_name}")


def get_file_data(file_path):
    with open(file_path) as f:
        csvv = csv.reader(f, delimiter="\t")
        value = list(csvv)[0][0]
        if value == 'NA':
            return None
        return float(value)


def get_all_bests(base_folder, output_folder, output_gedit=None):
    #algos = {"kl": False, "rmse": False, "pearson": True}
    algos = {"rmse": False}
    signs = os.listdir(base_folder)
    for sign in signs:
        sign_folder = f"{base_folder}/{sign}"
        mixes = os.listdir(sign_folder)
        for mix in mixes:
            mix_path = f"{sign_folder}/{mix}"
            for algo in algos:
                if output_gedit is None:
                    get_best_file_in_folder(mix_path, output_folder, algo, algos[algo])
                else:
                    get_best_file_in_folder(mix_path, output_folder, algo, algos[algo], is_gedit=False)
                    get_best_file_in_folder(mix_path, output_gedit, algo, algos[algo], is_gedit=True)


def get_all_bests2(base_folder, output_folder, output_gedit=None):
    algos = {"kl": False, "rmse": False, "pearson": True}
    signs = os.listdir(base_folder)
    for sign in signs:
        sign_folder = f"{base_folder}/{sign}"
        output_folder_sign = f"{output_folder}/{sign}"
        if not isdir(output_folder_sign):
            os.mkdir(output_folder_sign)
        for algo in algos:
            if output_gedit is None:
                get_best_file_in_folder(sign_folder, output_folder, algo, algos[algo])
            else:
                output_gedit_folder_sign = f"{output_gedit}/{sign}"
                if not isdir(output_gedit_folder_sign):
                    os.mkdir(output_gedit_folder_sign)

                get_best_file_in_folder(sign_folder, output_folder_sign, algo, algos[algo], is_gedit=False)
                get_best_file_in_folder(sign_folder, output_gedit_folder_sign, algo, algos[algo], is_gedit=True)


if __name__ == '__main__':
    # base_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/" \
    #             "Eran/benchmarking_results/Nmf-results"
    # main(base_folder)
    # base_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/" \
    #               "Eran/benchmarking_results/output_folder/"
    # output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/" \
    #                 "Eran/benchmarking_results/Best3"

    base_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/" \
                  "Eran/results_colab/results-elasticNet/res"
    output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/" \
                    "Eran/results_colab/results-elasticNet/best_values"
    output_folder_gedit = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/" \
                    "Eran/results_colab/results-elasticNet/best_values_gedit"

    base_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/25.2_model/output_folder-3/"
    output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/25.2_model/3-compare/"
    get_all_bests(base_folder, output_folder)


    #get_all_bests2(base_folder, output_folder, output_folder_gedit)
    # get_best_file_in_folder(base_folder, output_folder, "rmse")
