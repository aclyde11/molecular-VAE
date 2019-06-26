import torch
import torch.nn as nn
import torch.nn.functional as F


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

class BindingModel(nn.Module):
    def __init__(self, z_size=128):
        super().__init__()
        self.binding_model = nn.Sequential(
            nn.Linear(z_size, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.binding_model(x )

class VAE(nn.Module):
    def __init__(self, vocab):
        super().__init__()

        q_cell = "gru"
        q_bidir = False
        q_d_h = 256
        q_n_layers = 1
        q_dropout=0.05
        d_cell = 'gru'
        d_n_layers = 3
        d_dropout = 0.2
        self.d_z = 188
        d_z = self.d_z
        d_d_h=512

        self.vocabulary = vocab
        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(vocab.vectors)

        # Encoder
        if q_cell == 'gru':
            self.encoder_rnn = nn.GRU(
                d_emb,
                q_d_h,
                num_layers=q_n_layers,
                batch_first=True,
                dropout=q_dropout if q_n_layers > 1 else 0,
                bidirectional=q_bidir
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )

        q_d_last = q_d_h * (2 if q_bidir else 1)
        self.q_mu = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d_z))
        self.q_logvar = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d_z))

        # Decoder
        if d_cell == 'gru':
            self.decoder_rnn = nn.GRU(
                 d_z,
                d_d_h,
                num_layers=d_n_layers,
                batch_first=True,
                dropout=d_dropout if d_n_layers > 1 else 0
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of the ('gru',)"
            )

        self.decoder_lat = nn.Linear(d_z, d_d_h)
        self.decoder_fc = nn.Linear(d_d_h, n_vocab)


        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.x_emb,
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_lat,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])

        self.conv_1 = ConvSELU(40, 9, kernel_size=18)
        self.conv_2 = ConvSELU(9, 9, kernel_size=18)
        self.conv_3 = ConvSELU(9, 11, kernel_size=18)
        self.compacter = nn.Sequential(nn.Linear(561, 256), nn.ReLU())

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def forward(self, x, padded_x):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        x_t = torch.stack(x, dim=-1).cuda().long().permute((1, 0))
        x = self.x_emb(x_t)
        # Encoder: x -> z, kl_loss
        z, kl_loss, logvar = self.forward_encoder(x)

        # Decoder: x, z -> recon_loss
        recon_loss, x, y = self.forward_decoder(x_t, z)

        return kl_loss, recon_loss, z, logvar, x, y

    def forward_encoder(self, x):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = x.permute((0, 2, 1))
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        h = x.view(x.shape[0], -1)
        h = self.compacter(h)

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return z, kl_loss, logvar

    def forward_decoder(self, x, z):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        z_0 = z.unsqueeze(1).repeat(1, 102, 1)
        print(z_0.shape)

        # x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
        #                                             batch_first=True)
        x_input = z_0

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

        # output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        print(y.shape)
        print(x.shape)
        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )

        return recon_loss, x, y

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.d_z,
                         device=self.x_emb.weight.device)
        #
        # return torch.zeros((n_batch, self.d_z), device=self.x_emb.weight.device)


    def sample(self, n_batch, max_len=100, z=None, temp=1.0):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.x_emb.weight.device)
            z_0 = z.unsqueeze(1)

            # Initial values
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch,
                                                                    max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8,
                                   device=self.device)

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x], z