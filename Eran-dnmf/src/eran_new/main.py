import os
from pathlib import Path

import torch

from eran_new.dnmf_config import DnmfRange
from eran_new.train import DnmfConfig, train_manager


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
    wo_options = ["algo"]
    num_layers_options = [4, 5, 7]
    unsupervised_lr = 0.005
    supervised_trains = [50000, 80000]
    total_sigs = [75, 50]
    config_range = DnmfRange(wo_options, unsupervised_lr)

    refs = []
    for filename in os.listdir(ref_folder):
        refs.append(os.path.join(ref_folder, filename))
    mixes = []
    for filename in os.listdir(mix_folder):
        mixes.append(os.path.join(mix_folder, filename))

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
                            True,
                            True,
                            "algo",
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
                        learner, loss34 = train_manager(config, config_range)
                        key = learner.config.full_str()
                        loss_dict[key] = loss34
                        learners[key] = learner
        best_learner_conf = min(loss_dict, key=loss_dict.get)
        best_learner = learners[best_learner_conf]



if __name__ == "__main__":
    output_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/17.5/2_results/"
    ref_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/ref_mat_5/"
    dist_path = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/24.6/TrueProportions/"
    mix_folder = "/Users/Eran/Documents/benchmarking-transcriptomics-deconvolution/Figure1/Eran/m/"
    run_main(ref_folder, output_folder, mix_folder, dist_path)
