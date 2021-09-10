import torch
import torch.nn as nn
from torch.autograd import Variable


class AE1(nn.Module):
    def __init__(self, dim_data, dim_latents):
        super(AE1, self).__init__()
        self.encoder_mu = nn.Linear(dim_data, dim_latents)
        self.encoder_logvar = nn.Linear(dim_data, dim_latents)
        self.decoder = nn.Linear(dim_latents, dim_data)
        torch.nn.init.xavier_uniform_(self.encoder_mu.weight)
        torch.nn.init.constant_(self.encoder_mu.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.encoder_logvar.weight)
        torch.nn.init.constant_(self.encoder_logvar.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.constant_(self.decoder.bias, 0.0)

    def forward(self, x):
        _, _, z = self.get_z_half(x)
        x_recon = torch.sigmoid(self.decoder(z))
        return x_recon

    def get_z_half(self, x):
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        z = reparametrize(mu, logvar)
        return mu, logvar, z


class AE2(nn.Module):
    def __init__(self, dim_data, dim_latents):
        super(AE2, self).__init__()
        self.encoder_mu = nn.Linear(dim_data, dim_latents)
        self.encoder_logvar = nn.Linear(dim_data, dim_latents)
        self.decoder = nn.Linear(dim_latents, dim_data)
        torch.nn.init.xavier_uniform_(self.encoder_mu.weight)
        torch.nn.init.constant_(self.encoder_mu.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.encoder_logvar.weight)
        torch.nn.init.constant_(self.encoder_logvar.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.constant_(self.decoder.bias, 0.0)

    def forward(self, x):
        _, _, z = self.get_z_half(x)
        x_recon = torch.sigmoid(self.decoder(z))
        return x_recon

    def get_z_half(self, x):
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        z = reparametrize(mu, logvar)
        return mu, logvar, z


class Fusion(nn.Module):
    def __init__(self, h_dim, dim_latents, batch_size, is_cuda=False):
        super(Fusion, self).__init__()
        self.is_cuda = is_cuda
        self.h_dim = h_dim
        self.batch_size = batch_size
        self.dg1_encoder_mu = nn.Linear(h_dim, h_dim)
        self.dg1_encoder_logvar = nn.Linear(h_dim, h_dim)
        self.dg2_encoder_mu = nn.Linear(h_dim, h_dim)
        self.dg2_encoder_logvar = nn.Linear(h_dim, h_dim)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(dim_latents, h_dim))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(dim_latents, h_dim))
        self.experts = ProductOfExperts()
        torch.nn.init.xavier_uniform_(self.dg1_encoder_mu.weight)
        torch.nn.init.constant_(self.dg1_encoder_mu.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.dg1_encoder_logvar.weight)
        torch.nn.init.constant_(self.dg1_encoder_logvar.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.dg2_encoder_mu.weight)
        torch.nn.init.constant_(self.dg2_encoder_mu.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.dg2_encoder_logvar.weight)
        torch.nn.init.constant_(self.dg2_encoder_logvar.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.w1)
        torch.nn.init.xavier_uniform_(self.w2)

    def forward(self, z1, z2):
        h_mean, h_logvar = self.get_h(z1, z2)
        h = reparametrize(h_mean, h_logvar)
        g1 = h.matmul(self.w1.T)
        g2 = h.matmul(self.w2.T)
        return g1, g2, h_mean, h_logvar

    def get_h(self, z1, z2):
        h1 = z1.matmul(self.w1)
        h2 = z2.matmul(self.w2)
        mu, logvar = prior_expert((1, self.batch_size, self.h_dim), use_cuda=self.is_cuda)
        mu = torch.cat((mu, self.dg1_encoder_mu(h1).unsqueeze(0)), dim=0)
        logvar = torch.cat((logvar, self.dg1_encoder_logvar(h1).unsqueeze(0)), dim=0)
        mu = torch.cat((mu, self.dg2_encoder_mu(h2).unsqueeze(0)), dim=0)
        logvar = torch.cat((logvar, self.dg2_encoder_logvar(h2).unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)

        return mu, logvar

    def get_g(self, h_mean, h_logvar):
        vvv = reparametrize(h_mean, h_logvar)
        g1 = vvv.matmul(self.w1.T)
        g2 = vvv.matmul(self.w2.T)
        return g1, g2


def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)


class Discrimiter(nn.Module):
    def __init__(self, n_latents=200):
        super(Discrimiter, self).__init__()
        self.fc1 = nn.Linear(n_latents, 1)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x))


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar