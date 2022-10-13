import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch import nn

from eran_new.data_formatter import format_dataframe
from eran_new.train import _preprocess_dataframes
from eran_new.train_utils import _tensoring

TRUE_PATH_BASE_NEW = Path(
    "C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\24.6\\TrueProportionsNew"
)
TRUE_PATH_BASE = Path(
    "C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\24.6\\TrueProportions"
)
MIX_BASE = Path("C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\Mixes")
REF_BASE = Path("C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\RefMats")
GRAPH_SIZE = 1


def _get_algo_frame(algo_path: Path, true_prop: DataFrame, use_true_prop: bool) -> DataFrame:
    if not use_true_prop:
        return pd.read_csv(algo_path, sep="\t", index_col=0)
    algo_frame = pd.read_csv(algo_path, sep="\t", header=None)
    algo_frame.columns = true_prop.columns
    return algo_frame


def get_folder_graphs(
    path: Path, use_true_prop: bool, algo_name: str, graph_size: int = GRAPH_SIZE, save_normalize_graph: bool = False
) -> List[str]:
    dataset = path.name
    true_prop = TRUE_PATH_BASE / f"TrueProps{dataset}"  # {dataset.split('_')[0]}NormMix.tsv"
    mix_path = MIX_BASE / dataset
    criterion = nn.MSELoss(reduction="mean")
    loss_arr = []
    names_arr = []
    result_dict = {}
    unsup_result_dict = {}
    refs = [x for x in list(REF_BASE.iterdir())]
    final_ref = None
    for algorithm_name in os.listdir(path):
        for ref in refs:
            ref_name = ref.name.split(".")[0].lower()
            if ref_name in algorithm_name.lower():
                final_ref = ref
        assert final_ref is not None

        true_prop_pandas = pd.read_csv(true_prop, sep="\t", index_col=0)
        true_prop_pandas.index = true_prop_pandas.index.astype(str).str.upper()

        algorithm = str(algorithm_name)
        algo_pandas = _get_algo_frame(path / algorithm, true_prop_pandas, use_true_prop)

        ref_panda = pd.read_csv(REF_BASE / final_ref, sep="\t", index_col=0)
        ref_panda.index = ref_panda.index.str.upper()

        mix_panda_original = pd.read_csv(mix_path, sep="\t", index_col=0)
        mix_panda_original.index = mix_panda_original.index.str.upper()
        mix_panda_original.columns = mix_panda_original.columns.str.upper()

        ref_panda_gedit, mix_panda, _, _ = _preprocess_dataframes(ref_panda, mix_panda_original, true_prop_pandas, 50)

        algo_pandas, true_prop_pandas, _ = format_dataframe(algo_pandas, true_prop_pandas)
        algo_pandas.index = algo_pandas.index.astype(str).str.upper()
        true_prop_pandas_tensor = _tensoring(true_prop_pandas.values)
        algo_pandas = algo_pandas.loc[algo_pandas.index.intersection(true_prop_pandas.index)]
        assert algo_pandas.index.tolist() == true_prop_pandas.index.tolist()
        algo_tensor = _tensoring(algo_pandas.values)
        algo_pandas_normalized = algo_tensor / (torch.clamp(algo_tensor.sum(axis=1)[:, None], min=1e-12))
        assert algo_pandas_normalized.shape == true_prop_pandas_tensor.shape

        unsup_error = _tensoring(mix_panda) - _tensoring(ref_panda_gedit).matmul(algo_pandas_normalized.T)
        unsup_error = 0.5 * torch.pow(unsup_error, 2).sum()
        loss = torch.sqrt(criterion(algo_pandas_normalized, true_prop_pandas_tensor))
        loss_arr.append(float(loss))
        algorithm = algo_name + "\n" + algorithm
        algorithm = (
            algorithm.replace(",", "\n")
            .replace("50_Entropy_0.0_ScaledRef", "")
            .replace("$", "\n")
            .replace("(", "\n")
            .replace("_algo_", "algo")
            .replace("_1_", "1")
            .replace("Config", "")
            .replace("supervised_train", "train_size")
            .replace(").tsv", "")
            .replace(".tsv", "\n")
            .replace(".txt", "\n")
            .replace(".csv", "\n")
            .replace("OUT", "")
            .replace("\n_\n", "")
            .replace("GEDIT_", "GEDIT\n")
            .replace("Mix_", "Mix\n")
            .replace("l1_regularization=0", "")
            .replace("l2_regularization=1", "")
        )
        names_arr.append(algorithm)
        while algorithm in result_dict:
            algorithm = algorithm + "\n a"
        result_dict[algorithm] = float(loss)
        unsup_result_dict[algorithm] = float(unsup_error)
        if save_normalize_graph:
            directory = path.parent.parent / "normalized_graphs" / path.name
            directory.mkdir(exist_ok=True, parents=True)
            normalize_graph_path = directory / str(algorithm_name)
            algo_pandas[:] = algo_pandas_normalized.tolist()
            algo_pandas.to_csv(normalize_graph_path, sep="\t")

    unsup_sorted_lists = sorted(unsup_result_dict.items(), key=lambda x: x[1])[:graph_size]
    best_key = unsup_sorted_lists[0][0]
    result = [(best_key, result_dict[best_key])]
    # sorted_lists = sorted(result_dict.items(), key=lambda x: x[1])[:graph_size]
    for i in range(len(result)):
        key, value = result[i]
        result[i] = (key.split("\n")[0], value)
    return result


def create_graph(loss_arr: List[float], names_arr: List[str], name: str, description: str) -> None:
    x = np.arange(len(names_arr))  # the label
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, loss_arr, width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Rmse error")
    ax.set_title(f"Rmse error of {name} {description}")
    ax.set_xticks(x, names_arr, fontsize=8)
    ax.legend()

    ax.bar_label(rects1, padding=3)

    fig.tight_layout()

    plt.show()


def main_graph(
    path: Path, algo_name: str, description: str = "", use_true_prop: bool = False, save_normalize_graph: bool = False
) -> None:
    sorted_lists = get_folder_graphs(path, use_true_prop, algo_name, GRAPH_SIZE, save_normalize_graph)
    names_arr = [t[0] for t in sorted_lists]
    loss_arr = [t[1] for t in sorted_lists]
    create_graph(loss_arr, names_arr, path.name, description)


if __name__ == "__main__":
    path = Path(
        "C:\\Users\\Eran\\Documents\\benchmarking-transcriptomics-deconvolution\\Figure1\\Eran\\24.6\\Stromal_lm22_gedit"
    )
    # main_graph(path)
