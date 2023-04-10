import os
import pickle
from pathlib import Path

import torch
from torch import nn, optim

from eran_new.dnmf_config import DnmfConfig
from eran_new.train import SUPERVISED_SPLIT, train_manager, UNSUPERVISED_PREFIX, _train_unsupervised
from eran_new.train_utils import _tensoring
from scipy.stats import pearsonr


def run_main(ref_folder, output_folder, mix_folder, dist_folder, index):
    torch.autograd.set_detect_anomaly(True)

    refs = []
    for filename in os.listdir(ref_folder):
        refs.append(os.path.join(ref_folder, filename))

    mixes = []
    for filename in os.listdir(mix_folder):
        mixes.append(os.path.join(mix_folder, filename))
    mix = mixes[index]
    for ref_name in refs:
        mix_p = Path(mix)
        dist_path = Path(dist_folder) / f"TrueProps{mix_p.name}"
        config = DnmfConfig(
            use_gedit=True,
            use_w0=True,
            w1_option="algo",
            output_folder=Path(output_folder),
            unsup_output_folder=Path(output_folder),
            ref_path=Path(ref_name),
            mix_path=mix_p,
            dist_path=dist_path,
            num_layers=4,
            supervised_train=3 * SUPERVISED_SPLIT,
            unsupervised_train=60000,
            rewrite_exists_output=True,
            l1_regularization=0,
            l2_regularization=0,
            total_sigs=50,
            lr=0.001,
        )
        print(f"Main, {config.full_str()}")
        train_manager(config)


def run_main2(ref_folder, output_folder, mix_folder, dist_folder, index, output_folder_for_results=None):
    if output_folder_for_results is None:
        output_folder_for_results = output_folder
    torch.autograd.set_detect_anomaly(True)
    best_pearson = 0
    refs = []
    for filename in os.listdir(ref_folder):
        refs.append(os.path.join(ref_folder, filename))
    mixes = []
    for filename in os.listdir(mix_folder):
        mixes.append(os.path.join(mix_folder, filename))

    mix = mixes[index]
    criterion = nn.MSELoss(reduction="mean")

    best_of_best_loss_ref = 200
    best_of_best_nmf_ref = None

    for ref_name in refs:
        best_of_best_loss = 200
        learner_best = None

        mix_p = Path(mix)
        dist_path = Path(dist_folder) / f"TrueProps{mix_p.name}"

        config = DnmfConfig(
            use_gedit=True,
            use_w0=True,
            w1_option="algo",
            output_folder=Path(output_folder),
            unsup_output_folder=Path(output_folder_for_results),
            ref_path=Path(ref_name),
            mix_path=mix_p,
            dist_path=dist_path,
            num_layers=4,
            supervised_train=3 * SUPERVISED_SPLIT,
            unsupervised_train=20000,
            rewrite_exists_output=True,
            l1_regularization=0,
            l2_regularization=0,
            total_sigs=50,
            lr=0.001,
        )
        print(config.full_str())
        unsupervised_path = Path(
            str(config.output_path) + UNSUPERVISED_PREFIX + "_AAAGENERATED-UNsup_new_loss_algo.pkl"
        )
        if unsupervised_path.is_file():
            with open(str(unsupervised_path), "rb") as input_file:
                checkpoint = pickle.load(input_file)
            loss = 1  # checkpoint["loss34"]
            if loss < best_of_best_loss:
                best_of_best_alg = checkpoint["deep_nmf"]
                learner_best = train_manager(config, False)
                learner_best.deep_nmf = best_of_best_alg
                learner_best.optimizer = optim.Adam(best_of_best_alg.parameters(), lr=config.lr)
        else:
            print(f"doesnt have {unsupervised_path} file")
        if learner_best is not None:
            unsupervised_train_size = 101
            learner_best.config.unsupervised_train = unsupervised_train_size

            print("############################")
            print("start unsupervised learner_best")
            _, out34, _, _, _, loss = _train_unsupervised(learner_best)
            dist_new = learner_best.dist_mix_i.copy()
            dist_new[:] = out34.cpu().detach().numpy()
            current_pearson = pearsonr(dist_new.to_numpy().flatten(), learner_best.dist_mix_i.to_numpy().flatten())[0]
            curr_err = criterion(_tensoring(dist_new.to_numpy()), _tensoring(learner_best.dist_mix_i.to_numpy()))
            print(current_pearson, curr_err)
            if current_pearson > best_pearson:
                best_pearson = current_pearson

            if loss < best_of_best_loss_ref:
                best_of_best_loss_ref = loss
                best_of_best_nmf_ref = learner_best, out34

    print(f"best on all best ref is: {best_of_best_nmf_ref[0].config.full_str()} loss {best_of_best_loss_ref}")
    dist_new = best_of_best_nmf_ref[0].dist_mix_i.copy()
    dist_new[:] = best_of_best_nmf_ref[1].cpu().detach().numpy()
    dist_new.to_csv(best_of_best_nmf_ref[0].config.unsup_output_path, sep="\t")
    return

"""
import
    install_requires=[
        "pandas",
        "numpy",
        "typing",
        "torch",
        "attr",
        "scipy",
        "pickle",
        "tqdm"
    ]
"""
if __name__ == '__main__':
    output_folder = "../util_folder/out"
    ref_folder = "../util_folder/signatures"
    mixes_folder = "../util_folder/all_mixes/"
    true_prop_folder = "../util_folder/TrueProportions/"
    output_folder_final = "../util_folder/out_final"
    index = 0
    run_main(ref_folder, output_folder, mixes_folder, true_prop_folder, index)
    run_main2(ref_folder, output_folder, mixes_folder, true_prop_folder, index, output_folder_final)
