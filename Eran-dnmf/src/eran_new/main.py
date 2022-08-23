import os
import pickle
from pathlib import Path

import torch

from eran_new.dnmf_config import UnsupervisedLearner
from eran_new.dnmf_config import UnsuperNetNew
from layers.unsuper_layer import UnsuperLayer

import pathlib

from eran_new.train import DnmfConfig, train_manager, handle_best


def main():
    output_folder = Path("/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/17.5/2_results")
    ref_path = Path(
        "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/ref_mat_3/BlueCodeV1.tsv"
    )
    mix_path = Path(
        "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/17.5/Nmf-Objects-2/PBMC1NormMix.tsv"
    )
    dist_path = Path(
        "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/TrueProportions/TruePropsPBMC1NormMix.tsv"
    )
    config = DnmfConfig(
        True, True, "11", output_folder=output_folder, ref_path=ref_path, mix_path=mix_path, dist_path=dist_path
    )
    # train_manager(config)


def run_main(ref_folder, output_folder, mix_folder, dist_folder):
    torch.autograd.set_detect_anomaly(True)
    num_layers_options = [4]
    supervised_trains = [49999]
    total_sigs = [200, 50]

    refs = []
    for filename in os.listdir(ref_folder):
        refs.append(os.path.join(ref_folder, filename))
    mixes = []
    for filename in os.listdir(mix_folder):
        mixes.append(os.path.join(mix_folder, filename))
    for mix in mixes:
        for ref_name in refs:
            mix_p = Path(mix)
            dist_path = Path(dist_folder) / f"TrueProps{mix_p.name}"
            for num_layers_option in num_layers_options:
                for supervised_train in supervised_trains:
                    for total_sig in total_sigs:
                        config = DnmfConfig(
                            use_gedit=True,
                            use_w0=True,
                            w1_option="algo",
                            output_folder=Path(output_folder),
                            ref_path=Path(ref_name),
                            mix_path=mix_p,
                            dist_path=dist_path,
                            num_layers=num_layers_option,
                            supervised_train=supervised_train,
                            unsupervised_train=25000,
                            rewrite_exists_output=False,
                            total_sigs=total_sig,
                            lr=0.005,
                        )
                        print(config.full_str())
                        learner, loss34 = train_manager(config)


def run_main2(ref_folder, output_folder, mix_folder, dist_folder):
    torch.autograd.set_detect_anomaly(True)
    num_layers_options = [4, 5, 7]
    supervised_trains = [50000, 80000]
    total_sigs = [75, 50]

    refs = []
    for filename in os.listdir(ref_folder):
        refs.append(os.path.join(ref_folder, filename))
    mixes = []
    for filename in os.listdir(mix_folder):
        mixes.append(os.path.join(mix_folder, filename))

    best_of_best_alg = None
    best_of_best_loss = 2
    best_alg = None
    best_loss = 2

    for mix in mixes:
        loss_dict = {}
        learners = {}
        for ref_name in refs:
            mix_p = Path(mix)
            dist_path = Path(dist_folder) / f"TrueProps{mix_p.name}"
            for num_layers_option in num_layers_options:
                for supervised_train in supervised_trains:
                    for total_sig in total_sigs:
                        config = DnmfConfig(
                            use_gedit=True,
                            use_w0=True,
                            w1_option="algo",
                            output_folder=Path(output_folder),
                            ref_path=Path(ref_name),
                            mix_path=mix_p,
                            dist_path=dist_path,
                            num_layers=num_layers_option,
                            supervised_train=4,
                            unsupervised_train=2,
                            rewrite_exists_output=False,
                            total_sigs=total_sig,
                            lr=0.005,
                        )
                        print(config.full_str())
                        unsupervised_path = Path(str(config.output_path) + "GENERATED-UNsup.pkl")
                        unsupervised_path_best = Path(str(config.output_path) + "GENERATED-UNsupB.pkl")
                        with open(str(unsupervised_path), "rb") as input_file:
                            checkpoint = pickle.load(input_file)
                        loss = checkpoint["loss34"]
                        if loss < best_loss:
                            best_loss = loss
                            best_alg = checkpoint["deep_nmf"]
                            learner_best1 = train_manager(config, False)
                            learner_best1.deep_nmf = best_alg
                        with open(str(unsupervised_path_best), "rb") as input_file:
                            checkpoint = pickle.load(input_file)
                        loss = checkpoint["loss34"]
                        if loss < best_of_best_loss:
                            best_of_best_loss = loss
                            best_of_best_alg = checkpoint["deep_nmf"]
                            learner_best = train_manager(config, False)
                            learner_best.deep_nmf = best_of_best_alg

    print(f"best is {learner_best1.config.full_str()}")
    print(f"best is {learner_best.config.full_str()}")
    handle_best(learner_best1)
    handle_best(learner_best)


if __name__ == "__main__1":
    output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/17.5/2_results/"
    ref_name = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/ref_mat_4/BlueCodeV1.tsv"
    mix_p = Path("/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/m/Quan.tsv")
    dist_path = Path(
        "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/TrueProportions/TruePropsQuan.tsv"
    )

    config = DnmfConfig(
        use_gedit=True,
        use_w0=True,
        w1_option="algo",
        output_folder=Path(output_folder),
        ref_path=Path(ref_name),
        mix_path=mix_p,
        dist_path=dist_path,
        num_layers=4,
        supervised_train=50000,
        unsupervised_train=25000,
        rewrite_exists_output=False,
        total_sigs=50,
        lr=0.005,
    )
    checkpoint_path = "/Users/Eran/Downloads/DnmfConfig(use_w0=True, w1_option='algo', num_layers=4, total_sigs=50, l1_regularization=0, l2_regularization=1, unsupervised_train=25000).tsvGENERATED1-UNsupB.pkl"
    with open(str(checkpoint_path), "rb") as input_file:
        # temp = pathlib.PosixPath
        # pathlib.PosixPath = pathlib.WindowsPath

        checkpoint = pickle.load(input_file)
        loss = checkpoint["loss34"]
        best_of_best_alg = checkpoint["deep_nmf"]
        learner_best = train_manager(config, False)
        learner_best.deep_nmf = best_of_best_alg
        handle_best(learner_best)


if __name__ == "__main__":
    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath
    output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/17.5/2_results/"
    ref_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/ref_mat_5/"
    dist_path = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/24.6/TrueProportions/"
    mix_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/m/"
    run_main(ref_folder, output_folder, mix_folder, dist_path)
