import numpy as np

import torch
import torch.nn as nn

from model.glow import build_glow_model
from model.encoder import Encoder, ReverseFiLM_Encoder

from utils.utils import plot_graph


def reparameterize(mu, logvar):
    eps = torch.randn_like(logvar)
    z = eps*logvar.exp()+mu
    return z


def log_normal_diag(x, mean, logvar, dim=1):
    log_norm = -0.5 * (logvar + ((x - mean) ** 2) * logvar.exp().reciprocal())
    return torch.sum(log_norm, dim)


def log_normal_standard(x, dim=1):
    log_norm = -0.5 * x * x
    return torch.sum(log_norm, dim)


def nll(z, log_det_pz, dim=1):
    zz = torch.sum(z**2, dim)
    neg_log_likeli = 0.5 * zz - log_det_pz
    return neg_log_likeli


class VisionFlow(nn.Module):
    def __init__(self, dim_value: int, dim_text: int, num_nodes_img: int, num_nodes_value: int, dim_img: int, img_size: int, dim_latent: int, with_rfilm=True, is_alternate=True, x_invSig=None, x_Mu=None, device="cpu"):
        super(VisionFlow, self).__init__()

        self.dim_text = dim_text
        self.dim_value = dim_value
        self.num_nodes_img = num_nodes_img
        self.num_nodes_value = num_nodes_value
        self.dim_latent = dim_latent

        self.encoder = Encoder(dim_img, img_size, dim_text, dim_latent) if not with_rfilm else ReverseFiLM_Encoder(dim_img, img_size, dim_text, dim_latent)
        self.img_flow = build_glow_model(dim_tot=self.dim_latent,
                                         dim_cond=(self.dim_text,),
                                         num_nodes=self.num_nodes_img,
                                         rnvp_clamp=2.5,
                                         x_invSig=None, x_Mu=None)
        self.value_flow = build_glow_model(dim_tot=self.dim_value,
                                           dim_cond=(self.dim_text, self.dim_latent),
                                           num_nodes=self.num_nodes_value,
                                           rnvp_clamp=2.5,
                                           x_invSig=x_invSig, x_Mu=x_Mu)
        # plot_graph(self.img_flow.node_list, path='./', filename="img_flow")
        # plot_graph(self.value_flow.node_list, path='./', filename="value_flow")
        self.device = device

    def forward(self, x, img, text, rev=False):
        if rev:
            return self.reverse(x, img, text)
        z_mu, z_logvar = self.encoder(img, text)
        z0 = reparameterize(z_mu, z_logvar)
        z1, log_det_pz = self.img_flow(z0, c=text, jac=True, rev=False)
        y, log_det_py = self.value_flow(x, c=(text, z1), jac=True, rev=False)
        return y, log_det_py, z_mu, z_logvar, z0, z1, log_det_pz

    def reverse(self, y, img, text):
        z_mu, z_logvar = self.encoder(img, text)
        z0 = reparameterize(z_mu, z_logvar)
        z1, _ = self.img_flow(z0, c=text, jac=True, rev=False)
        x, _ = self.value_flow(y, c=(text, z1), jac=True, rev=True)
        return x

    def loss(self, y, log_det_py, z_mu, z_logvar, z0, z1, log_det_pz):
        log_n_z0 = log_normal_diag(z0, z_mu, z_logvar).mean()
        log_n_z1 = log_normal_standard(z1).mean()
        cond_div = nll(z1, log_det_pz).mean()
        value_div = nll(y, log_det_py).mean()
        return value_div+cond_div-log_n_z0, log_n_z0, log_n_z1, cond_div, value_div


if __name__ == "__main__":
    batch_size = 2
    img_ch = 4
    dim_img = 3
    img_size = 224
    img = torch.randn([batch_size, img_ch*dim_img, img_size, img_size]).cuda()
    text = torch.randn([batch_size, 7]).cuda()
    value = torch.randn([batch_size, 7]).cuda()
    model = VisionFlow(7, 7, 2, img_ch*dim_img, img_size, 128).cuda()
    print(model.encoder)
    print(model.value_flow)

    for i in range(1000):
        y, log_det_py, z_mu, z_logvar, z0, z1, log_det_pz = model(value, img, text)
        loss, log_n_z0, log_n_z1, value_div = model.loss(y, log_det_py, z_mu, z_logvar, z0, z1, log_det_pz)
        print(loss.mean().item(), log_n_z0.mean().item(), log_n_z1.mean().item(), value_div.mean().item())
        loss.backward()
        # break
