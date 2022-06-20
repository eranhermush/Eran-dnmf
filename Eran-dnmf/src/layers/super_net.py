import torch.nn as nn
import torch

from layers.super_layer import SuperLayer


class SuperNet(nn.Module):
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features, L1=True, L2=True):
        super(SuperNet, self).__init__()
        if L1:
            lambda1 = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            lambda1 = nn.Parameter(torch.zeros(1), requires_grad=False)
        if L2:
            lambda2 = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            lambda2 = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList([SuperLayer(comp, features, lambda1, lambda2) for i in range(self.n_layers)])

    def forward(self, h, x):
        # sequencing the layers and forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h
