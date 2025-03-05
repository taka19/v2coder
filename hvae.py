import gaussian as G
import component as nn


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 2 * channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 1)

    def forward(self, inputs, condition=None):
        x = self.conv1(inputs)
        if condition is not None:
            x = x + condition
        x = nn.gtu(x)
        x = self.conv2(x)
        outputs = inputs + x
        return outputs

class PriorBlock(nn.Module):
    def __init__(self, channels, latent_channels, kernel_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, 2 * channels, kernel_size),
            nn.GTU(),
            nn.Conv1d(channels, 2 * latent_channels, 1),
        )

    def forward(self, x):
        h = self.block(x)
        mu, log_lambda = h.chunk(2, dim=1)
        return G.GaussParam(mu, log_lambda)

class DecoderBlock(nn.Module):
    def __init__(self, channels, latent_channels, kernel_size):
        super().__init__()
        self.latent_conv = nn.Sequential(
            nn.Conv1d(latent_channels, channels, kernel_size),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size),
            nn.GELU(),
            nn.Conv1d(channels, 2 * channels, 1),
        )
        self.res_block = ResBlock(channels, kernel_size)

    def forward(self, x, latent):
        x = self.res_block(x, condition=self.latent_conv(latent))
        return x

class GenerativeLayer(nn.Module):
    def __init__(self, channels, latent_channels, kernel_size):
        super().__init__()
        self.prior_block = PriorBlock(channels, latent_channels, kernel_size)
        self.decoder_block = DecoderBlock(channels, latent_channels, kernel_size)

class GenerativeBlock(nn.Module):
    def __init__(self, in_channels, channels, latent_channels,
                 kernel_size, dilations, num_layers, up_factor):
        super().__init__()
        self.num_layers = num_layers
        self.up = nn.Upsample(in_channels, channels, up_factor, 1)
        self.layers = nn.ModuleList([
            GenerativeLayer(channels, latent_channels, kernel_size)
            for i in range(num_layers)
        ])

class ReconstructiveLayer(nn.Module):
    def __init__(self, channels, latent_channels, kernel_size, dilations):
        super().__init__()
        self.N = N = len(dilations)
        self.in_conv = nn.Conv1d(channels, channels, 1)
        self.cond_conv = nn.Conv1d(channels, N * 2 * channels, kernel_size)
        self.blocks = nn.ModuleList([
            ResBlock(channels, kernel_size, dilation=dilation)
            for dilation in dilations
        ])
        self.out_conv = nn.Conv1d(channels, 2 * latent_channels, 1)

    def forward(self, decoder_feats, determ_feats):
        c = self.cond_conv(decoder_feats).chunk(self.N, dim=1)
        x = self.in_conv(determ_feats)
        for i in range(self.N):
            x = self.blocks[i](x, c[i])

        x = self.out_conv(x)
        mu, log_lambda = x.chunk(2, dim=1)
        return G.GaussParam(mu, log_lambda)

class ReconstructiveBlock(nn.Module):
    def __init__(self, channels, latent_channels, kernel_size, dilations, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            ReconstructiveLayer(channels, latent_channels, kernel_size, dilations)
            for i in range(num_layers)
        ])

class DeterministicBlock(nn.Sequential):
    def __init__(self, in_channels, channels, kernel_size, dilations, down_factor):
        super().__init__(
            nn.Downsample(in_channels, channels, down_factor, 1),
            *[
                ResBlock(channels, kernel_size, dilation=dilation)
                for dilation in dilations
            ],
        )

def generate(glayer, x, eps=None):
    p = glayer.prior_block(x)
    z = G.sample(p, eps=eps)
    x = glayer.decoder_block(x, z)
    s = {
        "prior": p,
        "latent": z,
    }
    return x, s

def reconstruct(glayer, rlayer, x, determ_feats, eps=None):
    r = rlayer(x, determ_feats)
    p = glayer.prior_block(x)
    q = G.multiply(r, p)
    z = G.sample(q, eps=eps)
    x = glayer.decoder_block(x, z)

    s = {
        "prior": p,
        "posterior": q,
        "latent": z,
    }
    return x, s

