from collections import OrderedDict

import numpy as np
import torch
from einops import repeat
from torch import nn

from model.layers import Dropsample, SqueezeExcitation, mlp


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


def MBConv(
    dim_in,
    dim_out,
    downsample,
    expansion_rate=4,
    shrinkage_rate=0.25,
    dropout=0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


class ReverseFiLM(nn.Module):
    num_beta_gamma = 2  # one scale and one bias

    def __init__(self, dim_text, film_input_dim, num_params_to_film, layers):
        super().__init__()
        self.dim_text = dim_text
        self.film_input_dim = film_input_dim
        self.num_params_to_film = num_params_to_film
        self.film_output_dim = self.num_beta_gamma * num_params_to_film

        self.mlp1 = mlp(1024, layers, self.film_input_dim, self.film_output_dim)
        self.mlp2 = mlp(256, layers, self.dim_text, self.dim_text)

    def forward(self, x, text):
        film_output = self.mlp1(x)
        beta, delta_gamma = torch.chunk(film_output, chunks=self.num_beta_gamma, dim=1)
        gamma = delta_gamma + 1.0
        text = text * gamma + beta
        return self.mlp2(text)


class ReverseFiLM_Encoder(nn.Module):
    def __init__(self, dim_img, img_size, dim_text, dim_latent):
        super(ReverseFiLM_Encoder, self).__init__()
        downsample_time = int(np.log2(img_size))
        for i in range(downsample_time):
            if img_size / 2 ** i > img_size // 2 ** i:
                downsample_time = i - 1
                break
        last_kernel_size = img_size / 2 ** downsample_time
        assert last_kernel_size == int(last_kernel_size)

        q_z_nn = nn.ModuleList()
        lateral_conv = nn.ModuleList()
        conv_mix = nn.ModuleList()
        dim_in = dim_img
        for i in range(downsample_time):
            q_z_nn.append(MBConv(dim_in, dim_in * 2, downsample=True))
            size = int(img_size/2**(i+1))
            lateral_conv.append(nn.Conv2d(dim_in * 2, 128, kernel_size=size))
            conv_mix.append(ReverseFiLM(dim_text, 128, dim_text, 4))
            dim_in *= 2
        q_z_nn.append(nn.Conv2d(dim_in, dim_in * int(last_kernel_size), kernel_size=int(last_kernel_size)))
        self.q_z_nn = q_z_nn
        self.lateral_conv = lateral_conv
        self.conv_mix = conv_mix

        dim_internel = dim_in * int(last_kernel_size)
        self.q_z_mean = nn.Linear(dim_internel, dim_latent)
        self.q_z_mean_mix = mlp(1024, 4, dim_latent + dim_text*2, dim_latent)

        self.q_z_var = nn.Sequential(
            nn.Linear(dim_internel, dim_latent),
            nn.Softplus(), )
        self.q_z_var_mix = nn.Sequential(
            mlp(1024, 4, dim_latent + dim_text*2, dim_latent),
            nn.Softplus(), )

    def forward(self, img, text):
        h = img
        h_text = text
        N, C, W, H = img.shape
        for conv, lateral, mix in zip(self.q_z_nn[:-1], self.lateral_conv, self.conv_mix):
            h = conv(h)
            l = lateral(h).reshape([N, -1])
            h_text = mix(l, h_text)
        # h = self.q_z_nn(img)
        h = self.q_z_nn[-1](h)
        h = h.view(h.size(0), -1)
        h = repeat(h, "1 d -> b d", b=text.shape[0])

        m = self.q_z_mean(h)
        m = torch.cat([m, h_text, text], dim=1)
        m = self.q_z_mean_mix(m)

        v = self.q_z_var(h)
        v = torch.cat([v, h_text, text], dim=1)
        v = self.q_z_var_mix(v)
        return m, v


class Encoder(nn.Module):
    def __init__(self, dim_img, img_size, dim_text, dim_latent):
        super(Encoder, self).__init__()
        downsample_time = int(np.log2(img_size))
        for i in range(downsample_time):
            if img_size / 2 ** i > img_size // 2 ** i:
                downsample_time = i - 1
                break
        last_kernel_size = img_size / 2 ** downsample_time
        assert last_kernel_size == int(last_kernel_size)

        q_z_nn = OrderedDict()
        dim_in = dim_img
        for i in range(downsample_time):
            q_z_nn[f"MBconv_{i}"] = MBConv(dim_in, dim_in * 2, downsample=True)
            dim_in = dim_in * 2
        q_z_nn[f"Conv_Final"] = nn.Conv2d(dim_in, dim_in * int(last_kernel_size), kernel_size=int(last_kernel_size))
        self.q_z_nn = nn.Sequential(q_z_nn)

        dim_internel = dim_in * int(last_kernel_size)
        self.q_z_mean = nn.Linear(dim_internel, dim_latent)
        self.q_z_mean_mix = mlp(1024, 4, dim_latent + dim_text, dim_latent)

        self.q_z_var = nn.Sequential(
            nn.Linear(dim_internel, dim_latent),
            nn.Softplus(), )
        self.q_z_var_mix = nn.Sequential(
            mlp(1024, 4, dim_latent + dim_text, dim_latent),
            nn.Softplus(), )

    def forward(self, img, text):
        h = self.q_z_nn(img)
        h = h.view(h.size(0), -1)
        h = repeat(h, "1 d -> b d", b=text.shape[0])

        m = self.q_z_mean(h)
        m = torch.cat([m, text], dim=1)
        m = self.q_z_mean_mix(m)

        v = self.q_z_var(h)
        v = torch.cat([v, text], dim=1)
        v = self.q_z_var_mix(v)
        return m, v
