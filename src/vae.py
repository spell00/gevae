import torch.nn.utils.prune

import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import warnings

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.utils.distributions import log_gaussian, log_standard_gaussian
from src.utils.flows import NormalizingFlows, HouseholderFlow
from src.utils.layers import GaussianSample

writer = SummaryWriter('/home/simon/runs', purge_step=0)
writer.flush()
warnings.filterwarnings('ignore')

DEVICE = 'cuda'


def random_init(m, init_func=torch.nn.init.orthogonal_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


class Autoencoder(torch.nn.Module):
    def __init__(self,
                 z_dim,
                 batchnorm,
                 in_features=872,
                 hidden_size=128,
                 activation=torch.nn.ReLU,
                 flow_type="nf",
                 n_flows=10,
                 variational=True
                 ):
        super(Autoencoder, self).__init__()

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.variational = variational

        self.zs = torch.empty(size=(0, z_dim))

        self.device = device
        self.bns = []
        self.bns_deconv = []
        self.GaussianSample = GaussianSample(z_dim, z_dim)
        self.activation = activation()

        self.epoch = 0
        self.global_step = 0

        self.batchnorm = batchnorm
        self.a_dim = None
        self.dense1 = torch.nn.utils.weight_norm(torch.nn.Linear(in_features=in_features, out_features=hidden_size))
        self.dense11 = torch.nn.utils.weight_norm(torch.nn.Linear(in_features=hidden_size, out_features=z_dim))

        self.dense2 = torch.nn.utils.weight_norm(torch.nn.Linear(in_features=hidden_size, out_features=in_features))
        self.dense21 = torch.nn.utils.weight_norm(torch.nn.Linear(in_features=z_dim, out_features=hidden_size))
        self.dense1_bn = nn.BatchNorm1d(num_features=hidden_size)
        self.dense11_bn = nn.BatchNorm1d(num_features=z_dim)
        self.dense2_bn = nn.BatchNorm1d(num_features=in_features)
        self.dense21_bn = nn.BatchNorm1d(num_features=hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.bns = nn.ModuleList(self.bns)
        self.bns_deconv = nn.ModuleList(self.bns_deconv)
        self.flow_type = flow_type
        self.n_flows = n_flows
        if self.flow_type == "nf":
            self.flow = NormalizingFlows(in_features=[z_dim], n_flows=n_flows)
        if self.flow_type == "hf":
            self.flow = HouseholderFlow(in_features=[z_dim], auxiliary=False, n_flows=n_flows, h_last_dim=in_features)

    def random_init(self, init_func=torch.nn.init.kaiming_uniform_):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, h_last=None, p_param=None):
        if len(z.shape) == 1:
            z = z.view(1, -1)
        if (self.flow_type == "nf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z.type(torch.FloatTensor).to(DEVICE),
                              mu.type(torch.FloatTensor).to(DEVICE),
                              log_var.type(torch.FloatTensor).to(DEVICE)) - sum(log_det_z)
            z = f_z
        elif (self.flow_type == "iaf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z, h_last)
            qz = log_gaussian(z.type(torch.FloatTensor).to(DEVICE),
                              mu.type(torch.FloatTensor).to(DEVICE),
                              log_var.type(torch.FloatTensor).to(DEVICE)) - sum(log_det_z)
            z = f_z
        elif (self.flow_type in ['hf', 'ccliniaf']) and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z = self.flow(z, h_last)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        elif self.flow_type in ["o-sylvester", "h-sylvester", "t-sylvester"] and self.n_flows > 0:
            mu, log_var, r1, r2, q_ortho, b = q_param
            f_z = self.flow(z, r1, r2, q_ortho, b)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        else:
            (mu, log_var) = q_param
            qz = log_gaussian(z.type(torch.FloatTensor).to(DEVICE),
                              mu.type(torch.FloatTensor).to(DEVICE),
                              log_var.type(torch.FloatTensor).to(DEVICE))
        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z.type(torch.FloatTensor).to(DEVICE), mu, torch.exp(log_var.type(torch.FloatTensor)))

        kl = qz - pz

        return kl, z

    def encoder(self, z):
        z = self.dense1(z)
        z = self.activation(z)
        if self.batchnorm:
            if z.shape[0] != 1:
                z = self.dense1_bn(z)
        z = self.dropout(z)

        z = self.dense11(z)
        z = self.activation(z)
        if self.batchnorm:
            if z.shape[0] != 1:
                z = self.dense11_bn(z)
        z = self.dropout(z)
        return z

    def decoder(self, z):
        z = self.dense21(z)
        z = self.activation(z)
        if self.batchnorm:
            if z.shape[0] != 1:
                z = self.dense21_bn(z)
        z = self.dropout(z)

        z = self.dense2(z)
        z = self.activation(z)
        if self.batchnorm:
            if z.shape[0] != 1:
                z = self.dense2_bn(z)
        z = self.dropout(z)

        z = torch.sigmoid(z)
        return z

    def forward(self, x):
        z0 = self.encoder(x)
        z, mu, log_var = self.GaussianSample(z0)
        writer.add_histogram('z_gaussian', z, self.epoch)

        # Kullback-Leibler Divergence
        if self.variational:
            kl, z = self._kld(z, (mu, log_var), x)
        else:
            kl = 0
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        writer.add_histogram('z', z, self.epoch)
        self.zs = torch.cat((self.zs, z.detach().cpu()))
        rec = self.decoder(z)

        # kl = 0.01 * kl
        del mu, log_var, z0
        return rec, kl, z

    def sample(self, z, y=None):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
