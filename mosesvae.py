import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=110):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device='cuda')

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


        d_cell = 'gru'
        d_n_layers = 3
        d_dropout = 0.2
        self.d_z = 256
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
        self.encoder_rnn = nn.Sequential(
            ConvSELU(106, 64, kernel_size=9),
            ConvSELU(64, 48, kernel_size=9),
            ConvSELU(48, 32, kernel_size=11),
            ConvSELU(32, 10, kernel_size=11),
        )

        self.flatten = nn.Sequential(nn.Linear(740, 512, bias=True),
                      SELU(inplace=True))

        self.q_mu = nn.Sequential( nn.Linear(512, d_z, bias=True))
        self.q_logvar = nn.Sequential( nn.Linear(512, d_z, bias=True))

        # Decoder
        if d_cell == 'gru':
            self.decoder_rnn = nn.GRU(
                d_emb + d_z,
                d_d_h,
                num_layers=d_n_layers,
                batch_first=True,
                dropout=d_dropout if d_n_layers > 1 else 0
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of the ('gru',)"
            )

        self.decoder_lat = nn.Linear(d_z, d_d_h, bias=True)
        self.decoder_fc = nn.Linear(d_d_h, n_vocab, bias=True)


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

    def forward(self, x):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss, logvar = self.forward_encoder(x)

        # Decoder: x, z -> recon_loss
        recon_loss, x, y = self.forward_decoder(x, z)

        return kl_loss, recon_loss, z, logvar, x, y

    def forward_encoder(self, x):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)
        output, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=110)
        output=output.permute((0, 2, 1))
        x = self.encoder_rnn(output).view(output.shape[0], -1)
        x = self.flatten(x)


        mu, logvar = self.q_mu(x), self.q_logvar(x)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return z, kl_loss, logvar

    #
    # def decoder_step(self, input_tensor, target_tensor, z):
    #     import random
    #     decoder = AttnDecoderRNN(hidden_size=512, output_size=100)
    #     input_length = input_tensor.size(0)
    #     target_length = target_tensor.size(0)
    #
    #     x_emb = self.x_emb(input_tensor)
    #     encoder_outputs = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
    #
    #     decoder_input = torch.tensor([[self.eos]], device='cuda')
    #
    #     decoder_hidden = encoder_outputs
    #
    #     teacher_forcing_ratio = 0.5
    #     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #
    #     if use_teacher_forcing:
    #         # Teacher forcing: Feed the target as the next input
    #         for di in range(target_length):
    #             decoder_output, decoder_hidden, decoder_attention = decoder(
    #                 decoder_input, decoder_hidden, encoder_outputs)
    #             loss += criterion(decoder_output, target_tensor[di])
    #             decoder_input = target_tensor[di]  # Teacher forcing
    #
    #     else:
    #         # Without teacher forcing: use its own predictions as the next input
    #         for di in range(target_length):
    #             decoder_output, decoder_hidden, decoder_attention = decoder(
    #                 decoder_input, decoder_hidden, encoder_outputs)
    #             topv, topi = decoder_output.topk(1)
    #             decoder_input = topi.squeeze().detach()  # detach from history as input
    #
    #             loss += criterion(decoder_output, target_tensor[di])
    #             if decoder_input.item() == self.eos:
    #                 break

    def forward_decoder(self, x, z):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)

        if random.random() < 0.5:
            print("regular")
            x_emb = self.x_emb(x)
        else:
            print("Special")
            w = torch.tensor(self.bos, device=self.device).repeat(x.shape[0])
            x_emb = self.x_emb(w).unsqueeze(1)
        print(x_emb.shape)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

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