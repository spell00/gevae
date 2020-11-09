import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils.utils import random_init

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

class ResBlock(nn.Module):
    def __init__(self, in_hidden, hidden, activation=nn.ReLU, device='cuda'):
        super().__init__()
        self.activation = activation()
        self.device = device
        self.linear1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_hidden, hidden)),
        )
        self.linear2 = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.utils.weight_norm(nn.Linear(hidden, in_hidden)),
        )
        self.shortcut = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_hidden, hidden)),
            nn.BatchNorm1d(in_hidden)
        )
        self.linear1.apply(random_init).to(DEVICE)
        self.linear2.apply(random_init).to(DEVICE)

    def forward(self, x):
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparameterize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparameterize(mu, log_var), mu, log_var

