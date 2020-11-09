import torch
from torch import nn


def random_init(m, init_func=torch.nn.init.kaiming_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
