from pathlib import Path
from typing import List

import torch
from attr import define, field
from pandas import DataFrame
from torch import tensor
from torch.optim import Optimizer

from layers.unsuper_net_new import UnsuperNetNew


@define
class DnmfConfig:
    use_gedit: bool = field(repr=False)
    use_w0: bool = field()
    w1_option: str = field()
    mix_path: Path = field(repr=False)
    ref_path: Path = field(repr=False)
    dist_path: Path = field(repr=False)
    output_folder: str = field(repr=False)

    num_layers: int = field(default=4)
    total_sigs: int = field(default=50)
    l1_regularization = field(default=0)
    l2_regularization = field(default=1)
    supervised_train = field(default=80000, repr=False)
    unsupervised_train = field(default=10000, repr=False)
    lr: float = field(default=0.005, repr=False)
    dirichlet_alpha = field(default=1, repr=False)
    rewrite_exists_output: bool = field(default=True, repr=False)
    device = field(default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), repr=False)

    @property
    def output_path(self):
        return self.output_folder / self.mix_path.name / self.ref_path.name / str(self.use_gedit) / f"{str(self)}.tsv"

    def full_str(self):
        return f"{str(self)}, mix_path: {self.mix_path.name}, ref: {self.ref_path.name}, lr: {self.lr}, supervised_train: {self.supervised_train}, unsupervised_train: {self.unsupervised_train}"


@define
class DnmfRange:
    w1_options: List[int] = field()
    unsupervised_lr: float = field()

    def iterate_over_config(self, config: DnmfConfig):
        original_lr = config.lr
        config.lr = self.unsupervised_lr
        for w1 in self.w1_options:
            config.w1_option = w1
            yield config
        config.lr = original_lr


@define
class UnsupervisedLearner:
    config: DnmfConfig = field()
    deep_nmf: UnsuperNetNew = field()
    optimizer: Optimizer = field()
    mix_max: DataFrame = field()
    dist_mix_i: DataFrame = field()
    h_0: tensor = field()
