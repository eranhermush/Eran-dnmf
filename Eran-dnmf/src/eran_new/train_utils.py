from typing import Tuple

import numpy as np
import torch
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import nnls
from torch import tensor, optim
from torch.optim import Optimizer

from eran_new.dnmf_config import DnmfConfig
from layers.unsuper_net_new import UnsuperNetNew

EPSILON = torch.finfo(torch.float32).eps


def _init_dnmf(ref_mat: DataFrame, mix_max: DataFrame, config: DnmfConfig) -> Tuple[UnsuperNetNew, tensor, Optimizer]:
    features, n_components = ref_mat.shape
    samples, features = mix_max.shape
    deep_nmf = UnsuperNetNew(
        config.num_layers, n_components, features, config.l1_regularization, config.l2_regularization
    )
    for w in deep_nmf.parameters():
        w.data = (2 + torch.randn(w.data.shape, dtype=w.data.dtype)) * np.sqrt(2.0 / w.data.shape[0]) * 150
    h_0_train = _tensoring(
        np.asanyarray(
            [np.random.dirichlet(np.random.randint(1, 20, size=n_components)) for i in range(samples)], dtype=float
        )
    )
    if config.use_w0:
        deep_nmf_params = list(deep_nmf.parameters())
        for w_index in range(len(deep_nmf_params)):
            w = deep_nmf_params[w_index]
            if w_index == 0:
                w.data = _tensoring(np.dot(ref_mat.T, ref_mat))
                # w.requires_grad = False
            elif w_index == 1:
                w.data = _tensoring(ref_mat.values.T)
                # w.requires_grad = False

        h_0_train = [nnls(ref_mat.values, mix_max.values[kk])[0] for kk in range(len(mix_max))]
        h_0_train = _tensoring(np.asanyarray([d / sum(d) for d in h_0_train]))
    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=config.lr)
    return deep_nmf, h_0_train, optimizerADAM


def _tensoring(matrix: ndarray) -> tensor:
    """
    Convert numpy array to torch tensor
    """
    return torch.from_numpy(matrix).float()


def generate_dists(signature_data: DataFrame, std: float, alpha: int, mix_size: int) -> Tuple[tensor, tensor]:
    alpha_arr = np.full((signature_data.shape[0]), alpha, dtype=int)
    dist = np.asanyarray([np.random.dirichlet(alpha_arr) for _ in range(mix_size)], dtype=float)

    mix = dist.dot(signature_data)
    mix += np.random.normal(0, std, mix.shape)
    mix = np.maximum(mix, 0)
    return _tensoring(mix).T, _tensoring(dist)


def generate_new_w(Hi, Wi, V):
    denominator = torch.add(Wi.matmul(Hi).matmul(Hi.T), EPSILON)
    numerator = V.matmul(Hi.T)
    delta = torch.div(numerator, denominator)
    return torch.mul(delta, Wi)


def cost_tns(v, w, h, l_1=0, l_2=0):
    # util.cost_tns(data.v_train.tns,data.w.tns,data.h_train.tns)
    d = v - h.mm(w)
    return (0.5 * torch.pow(d, 2).sum() + l_1 * h.sum() + 0.5 * l_2 * torch.pow(h, 2).sum()) / h.shape[0]
