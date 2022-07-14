from layers.unsuper_layer import UnsuperLayer
import torch.nn as nn


class UnsuperNetNew(nn.Module):
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features, l_1=0, l_2=0):
        super(UnsuperNetNew, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList([UnsuperLayer(comp, features, l_1, l_2) for i in range(self.n_layers)])
        # self.softmax = nn.Softmax(1)

    def forward(self, h, x, weights=None):
        # sequencing the layers and forward pass through the network
        h_list = []
        weights_i = None
        for i, l in enumerate(self.deep_nmfs):
            if weights:
                weights_i = weights[f"deep_nmfs.{i}.fc1.weight"], weights[f"deep_nmfs.{i}.fc2.weight"]
            h = l(h, x, weights_i)
            h_list.append(h)
        return h, h_list
        # return self.softmax(h), h_list
