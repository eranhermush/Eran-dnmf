import torch.nn as nn
import torch.nn.functional as F
import torch

EPSILON = torch.finfo(torch.float32).eps


class UnsuperLayer(nn.Module):
    """
    Multiplicative update with Frobenius norm
    This can fit L1, L2 regularization.
    """

    def __init__(self, comp, features, l_1, l_2):
        super(UnsuperLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.l_1 = l_1
        self.l_2 = l_2
        self.fc1 = nn.Linear(comp, comp, bias=False)
        self.fc2 = nn.Linear(features, comp, bias=False)
        # self.softmax = nn.Softmax(1)
        # self.relu = nn.Tanh()

    def forward(self, y, x, weights=None):
        if weights is None:
            denominator = torch.add(self.fc1(y), self.l_2 * y + self.l_1 + EPSILON)
            numerator = self.fc2(x)
            delta = torch.div(numerator, denominator)
            return torch.mul(delta, y)
        fc1 = F.linear(y, weights[0])
        fc2 = F.linear(x, weights[1])
        denominator = torch.add(fc1, self.l_2 * y + self.l_1 + EPSILON)
        numerator = fc2
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)

        # return self.relu(torch.mul(delta, y)) + 1
        # return self.softmax(torch.mul(delta, y))
