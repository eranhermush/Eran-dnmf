import copy
import math
import os
import pickle
from datetime import datetime
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from scipy.optimize import nnls
from torch import nn, tensor, optim
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from eran_new.data_formatter import format_dataframe
from eran_new.data_frame_utils import get_shared_indexes
from eran_new.dnmf_config import DnmfConfig, DnmfRange, UnsupervisedLearner
from eran_new.gedit_new_version import run_gedit
from eran_new.train_utils import _init_dnmf, generate_dists, _tensoring, generate_new_w, cost_tns
from layers.unsuper_net_new import UnsuperNetNew


def train_manager(config: DnmfConfig, config_range: DnmfRange):
    if not _create_output_file(config):
        return
    ref_panda = pd.read_csv(config.ref_path, sep="\t", index_col=0)
    ref_panda.index = ref_panda.index.str.upper()

    mix_panda = pd.read_csv(config.mix_path, sep="\t", index_col=0)
    mix_panda.index = mix_panda.index.str.upper()

    dist_panda = pd.read_csv(config.dist_path, sep="\t", index_col=0)
    dist_panda.index = mix_panda.columns
    dist_panda.index = dist_panda.index.str.upper()

    ref_panda_gedit, mix_panda = _preprocess_dataframes(ref_panda, mix_panda, dist_panda, config)
    _, original_ref = get_shared_indexes(ref_panda_gedit, ref_panda)
    original_ref = format_dataframe(original_ref, dist_panda)

    _train_dnmf(mix_panda.T, ref_panda_gedit, original_ref, dist_panda, config, config_range)


def _handle_final_matrix(
    config: DnmfConfig, out: tensor, out34: tensor, deep_nmf: UnsuperNetNew, dist_panda: DataFrame
) -> None:
    dist_new = dist_panda.copy()
    criterion = nn.MSELoss(reduction="mean")
    loss = torch.sqrt(criterion(out, _tensoring(dist_panda.values).to(config.device)))
    loss34 = torch.sqrt(criterion(out34, _tensoring(dist_panda.values).to(config.device)))
    checkpoint = {"loss": loss, "loss34": loss34, "deep_nmf": deep_nmf, "config": config}
    print(f"final results: loss: {loss}, loss34: {loss34}")
    with open(str(config.output_path) + ".pkl", "wb") as f:
        pickle.dump(checkpoint, f)
    dist_new[:] = out.cpu().detach().numpy()
    dist_new.to_csv(config.output_path, sep="\t")

    dist_new[:] = out34.cpu().detach().numpy()
    dist_new.to_csv(str(config.output_path) + "34.tsv", sep="\t")

    print(f'Finish time: {datetime.now().strftime("%d-%m, %H:%M:%S")}')


def _create_output_file(config: DnmfConfig) -> bool:
    if not config.rewrite_exists_output and os.path.exists(config.output_path):
        return False

    if not os.path.isdir(config.output_path.parent):
        os.makedirs(config.output_path.parent, exist_ok=True)
    return True


def _preprocess_dataframes(
    ref_panda: DataFrame, mix_panda: DataFrame, dist_panda: DataFrame, config: DnmfConfig, use_all_genes: bool = False
) -> Tuple[DataFrame, DataFrame]:
    if config.use_gedit:
        ref_panda, mix_panda = run_gedit(mix_panda, ref_panda, config.total_sigs, use_all_genes)
    ref_panda = format_dataframe(ref_panda, dist_panda)
    return ref_panda, mix_panda


def _train_dnmf(
    mix_max: DataFrame, ref_mat: DataFrame, original_ref: DataFrame, dist_mix_i: DataFrame, config: DnmfConfig, config_range: DnmfRange
) -> None:
    print(f'start time: {datetime.now().strftime("%d-%m, %H:%M:%S")}')
    deep_nmf_original, h_0_train, optimizer = _init_dnmf(ref_mat, mix_max, config)
    deep_nmf_original.to(config.device)
    train_with_generated_data(original_ref, mix_max, dist_mix_i, deep_nmf_original, optimizer, config)
    for new_config in config_range.iterate_over_config(config):
        print(f'start unsupervised: : {datetime.now().strftime("%d-%m, %H:%M:%S")}')
        print(config.full_str())
        new_model = copy.deepcopy(deep_nmf_original)
        new_optimizer = optim.Adam(new_model.parameters(), lr=new_config.lr)

        learner = UnsupervisedLearner(new_config, new_model, new_optimizer, mix_max, dist_mix_i, h_0_train)
        out, out34, deep_nmf, best_34_obj, best_i = _train_unsupervised(learner)
        _handle_final_matrix(config, out, out34, deep_nmf, dist_mix_i)

        dist_new = dist_mix_i.copy()
        dist_new[:] = best_34_obj.cpu().detach().numpy()
        i = config.unsupervised_train
        config.unsupervised_train = best_i
        dist_new.to_csv(config.output_path, sep="\t")
        config.unsupervised_train = i


def _find_loss(
    config: DnmfConfig,
    deep_nmf_params: List[Parameter],
    mix_max: tensor,
    out: tensor,
    dist_mix_i_tensor: tensor,
    h_list: List[tensor],
    features: int,
) -> Tuple[tensor, tensor, tensor, tensor]:
    if config.w1_option == "1":
        dnmf_w = deep_nmf_params[1]
    elif config.w1_option == "last":
        dnmf_w = deep_nmf_params[len(deep_nmf_params) - 1]
    elif config.w1_option == "algo":
        w_i = deep_nmf_params[1].T
        mix_tensor = mix_max.T
        for j in range(len(h_list)):
            w_i = generate_new_w(h_list[j].T, w_i, mix_tensor)
        dnmf_w = nn.Softmax(1)(w_i.T)
    else:
        w_arrays = [nnls(out.data.numpy(), mix_max.T[f])[0] for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        dnmf_w = torch.from_numpy(nnls_w).float()

    loss = cost_tns(mix_max, dnmf_w, out, config.l1_regularization, config.l2_regularization)
    criterion = nn.MSELoss(reduction="mean")
    loss2 = torch.sqrt(criterion(out, dist_mix_i_tensor))

    out34 = out / (torch.clamp(out.sum(axis=1)[:, None], min=1e-12))
    loss34 = torch.sqrt(criterion(out34, dist_mix_i_tensor))

    return loss, loss2, loss34, out34


def _train_unsupervised(learner: UnsupervisedLearner) -> Tuple[tensor, tensor, UnsuperNetNew, tensor, int]:
    deep_nmf = learner.deep_nmf
    config = learner.config
    mix_max = _tensoring(learner.mix_max.values).to(config.device)
    optimizer = learner.optimizer

    samples, features = mix_max.shape
    dist_mix_i_tensor = _tensoring(learner.dist_mix_i.values).to(config.device)
    h_0 = learner.h_0.to(config.device)

    inputs = (h_0, mix_max)
    torch.autograd.set_detect_anomaly(True)
    out = np.ndarray([])
    out34 = np.ndarray([])
    best_34 = 2
    best_34_obj = out
    best_i = 0
    for i in range(config.unsupervised_train):
        out, h_list = deep_nmf(*inputs)
        deep_nmf_params = list(deep_nmf.parameters())
        loss, loss2, loss34, out34 = _find_loss(
            config, deep_nmf_params, mix_max, out, dist_mix_i_tensor, h_list, features
        )
        if i > 8000 and 0.125 < loss34.item() < 2:
            print("break: i = {i}")
            return out, out34, deep_nmf, best_34_obj, best_i
        if loss34 < best_34:
            best_34_obj = out
            best_34 = loss34
            best_i = i

        if i % 4000 == 0:
            print(f"i = {i}, loss =  {loss.item()},  loss criterion = {loss2} loss34 is {loss34} best {best_34}::{best_i}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)

    print(f"Finish: best {best_34}::{best_i}")
    return out, out34, deep_nmf, best_34_obj, best_i


def train_with_generated_data(
    ref_mat: DataFrame,
    mix_max: DataFrame,
    dist_matrix: DataFrame,
    deep_nmf: UnsuperNetNew,
    optimizer: Optimizer,
    config: DnmfConfig,
):
    rows, genes = mix_max.shape
    mix_dataframe = mix_max.copy()
    for train_index in range(config.supervised_train):
        generated_mix, generated_dist = generate_dists(ref_mat.T, train_index * 0.0001, config.dirichlet_alpha, rows)
        mix_data = generated_mix.to(config.device)
        #mix_dataframe[:] = generated_mix.detach().numpy().T
        #_, mix_frame = _preprocess_dataframes(ref_mat, mix_dataframe.T, dist_matrix, config, True)
        loss_values = train_supervised_one_sample(
            mix_data, generated_dist.to(config.device), deep_nmf, optimizer, config.device
        )
        if train_index % 8000 == 0:
            print(f"Train: Start train index: {train_index} with loss: {loss_values}")


def train_supervised_one_sample(
    mix_max: DataFrame, dist_mat: tensor, deep_nmf: UnsuperNetNew, optimizer: Optimizer, device
):
    n_h_rows, n_components = dist_mat.shape
    samples = mix_max.shape[0]
    h_0_train = _tensoring(
            np.random.dirichlet(np.random.randint(1, 15, size=n_components), samples)
    ).to(device)

    criterion = nn.MSELoss(reduction="mean")
    # Train the Network
    # out, _ = deep_nmf(h_0_train, _tensoring(mix_max.T.values).to(device))
    out, _ = deep_nmf(h_0_train, mix_max)
    loss = torch.sqrt(criterion(out, dist_mat))  # loss between predicted and truth
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for w in deep_nmf.parameters():
        w.data = w.clamp(min=0, max=math.inf)
    return loss.item()
