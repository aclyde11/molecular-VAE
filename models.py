import torch
import torch.nn as nn
from torch.autograd import Variable


class Flatten(nn.Module):

    def forward(self, x):
        size = x.size()  # read in N, C, H, W
        return x.view(size[0], -1)


class Repeat(nn.Module):

    def __init__(self, rep):
        super(Repeat, self).__init__()

        self.rep = rep

    def forward(self, x):
        size = tuple(x.size())
        size = (size[0], 1) + size[1:]
        x_expanded = x.view(*size)
        n = [1 for _ in size]
        n[1] = self.rep
        return x_expanded.repeat(*n)


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))

        return y


class SELU(nn.Module):

    def __init__(self, alpha=1.6732632423543772848170429916717,
                 scale=1.0507009873554804934193349852946, inplace=False):
        super(SELU, self).__init__()

        self.scale = scale
        self.elu = nn.ELU(alpha=alpha, inplace=inplace)

    def forward(self, x):
        return self.scale * self.elu(x)


def ConvSELU(i, o, kernel_size=3, padding=0, p=0.):
    model = [nn.Conv1d(i, o, kernel_size=kernel_size, padding=padding),
             SELU(inplace=True)
             ]
    if p > 0.:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)


class Lambda(nn.Module):

    def __init__(self, i=435, o=292, scale=1E-2):
        super(Lambda, self).__init__()

        self.scale = scale
        self.z_mean = nn.Linear(i, o)
        self.z_log_var = nn.Linear(i, o)

    def forward(self, x):
        self.mu = self.z_mean(x)
        self.log_v = self.z_log_var(x)
        eps = self.scale * Variable(torch.randn(*self.log_v.size())
                                    ).type_as(self.log_v)
        return self.mu + torch.exp(self.log_v / 2.) * eps, self.mu, self.log_v


class MolecularVAE(nn.Module):
    def __init__(self, i=120, o=292, c=35):
        super(MolecularVAE, self).__init__()

        self.encoder = MolEncoder(i=i, o=o, c=c)
        self.decoder = MolDecoder(i=o, o=i, c=c)

    def forward(self, x):
        x, mu, logvar = self.encoder(x)
        return self.decoder(x), mu, logvar


class MolEncoder(nn.Module):

    def __init__(self, i=120, o=292, c=35):
        super(MolEncoder, self).__init__()

        self.i = i

        self.conv_1 = ConvSELU(i, i, kernel_size=11)
        self.conv_2 = ConvSELU(i, 128, kernel_size=9)
        self.conv_3 = ConvSELU(128, 64, kernel_size=5)
        self.conv_4 = ConvSELU(64, 32, kernel_size=4)

        self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 5) * 32, 512),
                                     SELU(inplace=True))

        self.lmbd = Lambda(512, o)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = Flatten()(out)
        out = self.dense_1(out)

        return self.lmbd(out)

    def vae_loss(self, x_decoded_mean, x):
        z_mean, z_log_var = self.lmbd.mu, self.lmbd.log_v

        bce = nn.BCELoss(size_average=True)
        xent_loss = self.i * bce(x_decoded_mean, x.detach())
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. -
                                    torch.exp(z_log_var))

        return kl_loss + xent_loss


class MolDecoder(nn.Module):

    def __init__(self, i=292, o=120, c=35, h=501):
        super(MolDecoder, self).__init__()

        self.latent_input = nn.Sequential(nn.Linear(i, i),
                                          SELU(inplace=True))
        self.repeat_vector = Repeat(o)
        self.gru = nn.GRU(i, h, 4, batch_first=True)
        self.decoded_mean = TimeDistributed(nn.Sequential(nn.Linear(h, c),
                                                          nn.Softmax())
                                            )

    def forward(self, x):
        out = self.latent_input(x)
        out = self.repeat_vector(out)
        out, h = self.gru(out)
        return self.decoded_mean(out)
