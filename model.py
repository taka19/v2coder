from math import log
import numpy as np
import torch
import torch.nn.functional as F

import gaussian as G
import hvae
import component as nn


class GenerativeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_scales = len(cfg["hvae"]["num_layers"])
        self.num_layers = sum(cfg["hvae"]["num_layers"])

        in_channels = cfg["msg"]["channels"]
        c = cfg["hvae"]
        self.blocks = nn.ModuleList()
        for (
                channels,
                latent_channels,
                num_layers,
                up_factor,
        ) in zip(
            c["channels"],
            c["latent_channels"],
            c["num_layers"],
            [1] + c["up_factors"],
        ):
            self.blocks.append(
                hvae.GenerativeBlock(
                    in_channels, channels, latent_channels,
                    c["kernel_size"], c["dilations"], num_layers, up_factor,
                )
            )
            in_channels = channels

        self.to_wav = nn.Conv1d(channels, cfg["wav"]["channels"], 1)
        self.register_buffer("log_decoder_precision", torch.tensor(cfg["log_decoder_precision"]))

    def forward(self, msg):
        x = msg
        states = {}
        for i, gblock in enumerate(self.blocks):
            x = gblock.up(x)
            states[i] = []

            for j, glayer in enumerate(gblock.layers):
                x, s = hvae.generate(
                    glayer,
                    x,
                )
                states[i].append(s)

        mu = self.to_wav(x)
        p_wav = G.GaussParam(mu, torch.full_like(mu, self.log_decoder_precision))
        return p_wav

class InferenceModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_scales = len(cfg["hvae"]["num_layers"])

        c = cfg["hvae"]
        self.determ_blocks = nn.ModuleList()
        self.reconstructive_blocks = nn.ModuleList()
        for i, (
                in_channels,
                channels,
                latent_channels,
                num_layers,
                down_factor,
        ) in enumerate(zip(
            c["channels"][1:] + [cfg["wav"]["channels"]],
            c["channels"],
            c["latent_channels"],
            c["num_layers"],
            c["up_factors"] + [1],
        )):
            self.determ_blocks.append(
                hvae.DeterministicBlock(
                    in_channels, channels,
                    c["kernel_size"],
                    c["determ_blocks"]["dilations"],
                    down_factor,
                )
            )
            self.reconstructive_blocks.append(
                hvae.ReconstructiveBlock(
                    channels, latent_channels,
                    c["kernel_size"], c["dilations"], num_layers,
                )
            )

class LagrangeMultiplier(nn.Parameter):
    def __new__(cls, value=log(1.0e-8), shape=[], requires_grad=True):
        return super().__new__(cls, data=torch.full(shape, value), requires_grad=requires_grad)

class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.generative_model = GenerativeModel(cfg)
        self.inference_model = InferenceModel(cfg)

    def extract_deterministic_features(self, wav):
        blocks = self.inference_model.determ_blocks

        feats = {}
        h = wav
        for i in reversed(range(len(blocks))):
            h = feats[i] = blocks[i](h)

        return feats

    def reconstruct(self, msg, determ_feats):
        gmodel = self.generative_model
        imodel = self.inference_model

        skip_intval = gmodel.num_layers // 6
        x = msg
        skip_x = None
        states = {}
        layer_idx = 0
        for i, (gblock, rblock) in enumerate(zip(gmodel.blocks, imodel.reconstructive_blocks, strict=True)):
            x = gblock.up(x)
            if skip_x is not None:
                skip_x = gblock.up(skip_x)
            states[i] = []

            for j, (glayer, rlayer) in enumerate(zip(gblock.layers, rblock.layers, strict=True)):
                x, s = hvae.reconstruct(
                    glayer,
                    rlayer,
                    x,
                    determ_feats[i],
                )

                if (layer_idx + 1) % skip_intval == 0:
                    if skip_x is None:
                        skip_x = x
                    else:
                        skip_x = torch.cat([skip_x, x], dim=0)

                states[i].append(s)
                layer_idx += 1

        mu = gmodel.to_wav(x)
        p_wav = G.GaussParam(mu, gmodel.log_decoder_precision.expand_as(mu))
        mu = gmodel.to_wav(skip_x).view(-1, mu.size(0), mu.size(1), mu.size(2))
        p_wav_skip = G.GaussParam(mu, gmodel.log_decoder_precision.expand_as(mu))
        return p_wav, p_wav_skip, states

class Model(nn.ModuleDict):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__({
            "vae": VAE(cfg),
            "lagrange_multipliers": nn.ParameterList([
                nn.ParameterList([
                    LagrangeMultiplier() for _ in range(num_layers)
                ]) for num_layers in cfg["hvae"]["num_layers"]
            ]),
        })

    def get_max_rates(self, s, e):
        L = self.vae.generative_model.num_layers
        t = np.arange(L) / (L - 1)
        rates = s * (e / s) ** t
        return rates

    def forward(self, wav, msg, **hparams):
        wav_length = wav.size(2)

        determ_feats = self.vae.extract_deterministic_features(wav)
        p_wav, p_wav_skip, states = self.vae.reconstruct(msg, determ_feats)

        distortion = G.negative_log_prob1d(wav, p_wav, dim=(1, 2)).mean(dim=0)
        distortion_skip = G.negative_log_prob1d(wav.unsqueeze(0), p_wav_skip, dim=(2, 3)).mean(dim=1).sum(0)

        C = None
        if hparams["constraint"] is not None:
            C = wav_length * self.get_max_rates(**hparams["constraint"])

        cn_rate = rate = 0.0
        layer_idx = 0
        for i in range(len(states)):
            for j, s in enumerate(states[i]):
                p, q = s["prior"], s["posterior"]
                kld = G.kld1d(q, p, dim=(1, 2)).mean(dim=0)
                if C is not None:
                    cn_rate += self.lagrange_multipliers[i][j].exp() * (kld - C[layer_idx])
                rate += kld
                layer_idx += 1

        distortion, rate, distortion_skip, cn_rate = (
            x / wav_length for x in [distortion, rate, distortion_skip, cn_rate]
        )
        loss = distortion + rate + cn_rate + hparams["skip_weight"] * distortion_skip

        return loss, (distortion, rate, cn_rate), p_wav.mu

def remove_parametrizations(m):
    if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
        torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
