import math
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import nnls
from torch import nn, tensor
from torch.optim import Optimizer

from eran_new.data_formatter import format_dataframe
from eran_new.dnmf_config import DnmfConfig, DnmfRange, UnsupervisedLearner
from eran_new.gedit_new_version import run_gedit, _quantile_normalize, _normalize_zero_one
from eran_new.train_utils import _init_dnmf, generate_dists, _tensoring
from layers.unsuper_net_new import UnsuperNetNew


def train_manager(config: DnmfConfig, config_range: DnmfRange):
    if not _create_output_file(config):
        return
    ref_panda = pd.read_csv(config.ref_path, sep="\t", index_col=0)
    ref_panda.index = ref_panda.index.str.upper()

    mix_panda_original = pd.read_csv(config.mix_path, sep="\t", index_col=0)
    mix_panda_original.index = mix_panda_original.index.str.upper()

    dist_panda = pd.read_csv(config.dist_path, sep="\t", index_col=0)
    dist_panda.index = mix_panda_original.columns
    dist_panda.index = dist_panda.index.str.upper()

    ref_panda_gedit, mix_panda, indexes = _preprocess_dataframes(ref_panda, mix_panda_original, dist_panda, config)
    original_ref = format_dataframe(ref_panda, dist_panda)
    original_ref = original_ref.loc[indexes]
    original_ref = original_ref.sort_index(axis=0)

    learner, loss34 = _train_dnmf(mix_panda.T, ref_panda_gedit, original_ref.to_numpy(), dist_panda, config, config_range)
    return learner, loss34


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_panda = format_dataframe(ref_panda, dist_panda)
    ref_panda, mix_panda, indexes = run_gedit(mix_panda, ref_panda, config.total_sigs, use_all_genes)
    return ref_panda, mix_panda, indexes


def _train_dnmf(
    mix_max: ndarray,
    ref_mat: ndarray,
    original_ref: ndarray,
    dist_mix_i: DataFrame,
    config: DnmfConfig,
    config_range: DnmfRange,
) -> None:
    print(f'start time: {datetime.now().strftime("%d-%m, %H:%M:%S")}')
    deep_nmf_original, h_0_train, optimizer = _init_dnmf(ref_mat, mix_max, config)
    deep_nmf_original.to(config.device)
    original_ref = original_ref[(original_ref != 0).any(axis=1)]
    p = Path(str(config.output_path) + "GENERATED.pkl")
    if p.is_file():
        with open(str(p), "rb") as input_file:
            checkpoint = pickle.load(input_file)
            deep_nmf_original = checkpoint["deep_nmf"]
    else:
        train_with_generated_data(ref_mat, original_ref, mix_max, deep_nmf_original, optimizer, config)
        checkpoint = {"deep_nmf": deep_nmf_original, "config": config}
        with open(str(config.output_path) + "GENERATED.pkl", "wb") as f:
            pickle.dump(checkpoint, f)

    loss34 = train_with_generated_data_unsupervised(ref_mat, original_ref, mix_max, deep_nmf_original, optimizer, config)
    print(f'start unsupervised: : {datetime.now().strftime("%d-%m, %H:%M:%S")}')
    print(config.full_str())

    learner = UnsupervisedLearner(config, deep_nmf_original, optimizer, mix_max, dist_mix_i, h_0_train)
    return learner, loss34

def handle_best(learner: UnsupervisedLearner):
    config = learner.config
    dist_mix_i = learner.dist_mix_i
    out, out34, deep_nmf, best_34_obj, best_i = _train_unsupervised(learner)
    _handle_final_matrix(config, out, out34, deep_nmf, dist_mix_i)

    dist_new = dist_mix_i.copy()
    dist_new[:] = best_34_obj.cpu().detach().numpy()
    i = config.unsupervised_train
    config.unsupervised_train = best_i
    dist_new.to_csv(config.output_path, sep="\t")
    config.unsupervised_train = i


def _find_loss(
    out: tensor,
    dist_mix_i_tensor: tensor,
) -> Tuple[tensor, tensor, tensor]:
    # if config.w1_option == "1":
    #     dnmf_w = deep_nmf_params[1]
    # elif config.w1_option == "last":
    #     dnmf_w = deep_nmf_params[len(deep_nmf_params) - 1]
    # elif config.w1_option == "algo":
    #     w_i = deep_nmf_params[1].T
    #     mix_tensor = mix_max.T
    #     for j in range(len(h_list)):
    #         w_i = generate_new_w(h_list[j].T, w_i, mix_tensor)
    #     dnmf_w = nn.Softmax(1)(w_i.T)
    # else:
    #     w_arrays = [nnls(out.data.numpy(), mix_max.T[f])[0] for f in range(features)]
    #     nnls_w = np.stack(w_arrays, axis=-1)
    #     dnmf_w = torch.from_numpy(nnls_w).float()

    # loss = cost_tns(mix_max, dnmf_w, out, config.l1_regularization, config.l2_regularization)
    criterion = nn.MSELoss(reduction="mean")
    loss2 = torch.sqrt(criterion(out, dist_mix_i_tensor))

    out34 = out / (torch.clamp(out.sum(axis=1)[:, None], min=1e-12))
    loss34 = torch.sqrt(criterion(out34, dist_mix_i_tensor))

    return loss2, loss34, out34


def _train_unsupervised(learner: UnsupervisedLearner) -> Tuple[tensor, tensor, UnsuperNetNew, tensor, int]:
    deep_nmf = learner.deep_nmf
    config = learner.config
    mix_max = _tensoring(learner.mix_max).to(config.device)
    optimizer = learner.optimizer
    dist_mix_i_tensor = _tensoring(learner.dist_mix_i.to_numpy()).to(config.device)
    h_0 = learner.h_0.to(config.device)

    inputs = (h_0, mix_max)
    torch.autograd.set_detect_anomaly(True)
    out = ndarray([])
    out34 = ndarray([])
    best_34 = 2
    best_34_obj = out
    best_i = 0
    for i in range(config.unsupervised_train):
        out, loss = deep_nmf(*inputs)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(deep_nmf.parameters(), clip_value=1)
        optimizer.step()
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)

        with torch.no_grad():
            loss2, loss34, out34 = _find_loss(out, dist_mix_i_tensor)
        if i > 8000 and 0.125 < loss34.item() < 2:
            print(f"break: i = {i}")
            return out, out34, deep_nmf, best_34_obj, best_i
        if loss34 < best_34:
            best_34_obj = out
            best_34 = loss34
            best_i = i

        if i % 4000 == 0:
            print(
                f"i = {i}, loss =  {loss.item()},  loss criterion = {loss2} loss34 is {loss34} best {best_34}::{best_i}"
            )
            print(f'{datetime.now().strftime("%d-%m, %H:%M:%S")}')
    print(f"Finish: best {best_34}::{best_i}")
    return out, out34, deep_nmf, best_34_obj, best_i


def train_with_generated_data_unsupervised(
    ref_mat: ndarray,
    original_ref_mat: ndarray,
    mix_max: ndarray,
    deep_nmf: UnsuperNetNew,
    optimizer: Optimizer,
    config: DnmfConfig,
):
    torch.autograd.set_detect_anomaly(True)
    rows, genes = mix_max.shape
    best_34 = 2
    best_i = 0
    for train_index in range(config.supervised_train):
        generated_mix, generated_dist = generate_dists(
            original_ref_mat.T, train_index * 0.0001, config.dirichlet_alpha, rows
        )
        mix_max = _quantile_normalize(generated_mix, original_ref_mat)
        _, mix_frame = _normalize_zero_one(original_ref_mat, mix_max)

        generated_dist = _tensoring(generated_dist).to(config.device)
        mix_max = mix_max.T
        h_0_train = [nnls(ref_mat, mix_max[kk])[0] for kk in range(len(mix_max))]
        h_0_train = _tensoring(np.asanyarray([d / sum(d) for d in h_0_train])).to(config.device)
        mix_max = _tensoring(mix_max).to(config.device)
        out, loss = deep_nmf(h_0_train, mix_max)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(deep_nmf.parameters(), clip_value=1)
        optimizer.step()
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)

        with torch.no_grad():
            loss2, loss34, out34 = _find_loss(out, generated_dist)
        if train_index > 8000 and 0.125 < loss34.item() < 2:
            print(f"break: i = {train_index}")
            return loss34
        if loss34 < best_34:
            best_34 = loss34
            best_i = train_index

        if train_index % 4000 == 0:
            print(
                f"i = {train_index}, loss =  {loss.item()},  loss criterion = {loss2} loss34 is {loss34} best {best_34}::{best_i}"
            )
            print(f'{datetime.now().strftime("%d-%m, %H:%M:%S")}')
    return loss34


def train_with_generated_data(
    ref_mat: ndarray,
    original_ref_mat: ndarray,
    mix_max: ndarray,
    deep_nmf: UnsuperNetNew,
    optimizer: Optimizer,
    config: DnmfConfig,
):
    rows, genes = mix_max.shape
    for train_index in range(config.supervised_train):
        generated_mix, generated_dist = generate_dists(
            original_ref_mat.T, train_index * 0.0001, config.dirichlet_alpha, rows
        )
        # mix_dataframe[:] = generated_mix.detach().numpy().T
        mix_max = _quantile_normalize(generated_mix, original_ref_mat)
        _, mix_frame = _normalize_zero_one(original_ref_mat, mix_max)

        #mix_frame = _tensoring(mix_frame).to(config.device)
        generated_dist = _tensoring(generated_dist).to(config.device)

        loss_values = train_supervised_one_sample(
            ref_mat, mix_frame.T, generated_dist, deep_nmf, optimizer, config.device
        )
        if train_index % 8000 == 0:
            print(f"Train: Start train index: {train_index} with loss: {loss_values}")


def train_supervised_one_sample(
    ref_mat: ndarray, mix_max: ndarray, dist_mat: tensor, deep_nmf: UnsuperNetNew, optimizer: Optimizer, device
):
    n_h_rows, n_components = dist_mat.shape
    samples = mix_max.shape[0]
    # h_0_train = _tensoring(
    #         np.random.dirichlet(np.random.randint(1, 15, size=n_components), samples)
    # ).to(device)
    #r = mix_max.cpu().detach().numpy()
    h_0_train = [nnls(ref_mat, mix_max[kk])[0] for kk in range(len(mix_max))]
    # h_0_train = [nnls(ref_mat, mix_max[kk])[0] for kk in range(len(mix_max))]
    h_0_train = _tensoring(np.asanyarray([d / sum(d) for d in h_0_train])).to(device)

    criterion = nn.MSELoss(reduction="mean")
    # Train the Network
    # out, _ = deep_nmf(h_0_train, _tensoring(mix_max.T.values).to(device))
    mix_max = _tensoring(mix_max).to(device)
    out, _ = deep_nmf(h_0_train, mix_max)
    loss = torch.sqrt(criterion(out, dist_mat))  # loss between predicted and truth
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(deep_nmf.parameters(), clip_value=1)
    optimizer.step()
    for w in deep_nmf.parameters():
        w.data = w.clamp(min=0, max=math.inf)
    return loss.item()
