import torch
from torch import nn
from torch.nn import functional as F
from types_ import *


class VAE(nn.Module):

    def __init__(self,
                 n_genes: int = 3608,
                 latent_dim: int = 128,
                 n_labels: int = 4,

                 nhead=8,
                 dropout=0.1,
                 dim_feedforward=512,
                 hidden_dims: List = None,
                 hidden_dims_pre: List = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            # hidden_dims = [32, 64, 128, 256, 512]
            hidden_dims = [512,256, 256, 128]
        if hidden_dims_pre is None:
            hidden_dims_pre = [16, 8]

        # Build Encoder
        h_in = n_genes
        for h_dim in hidden_dims:

            modules.append(
                nn.Sequential(
                    nn.Linear(h_in, h_dim),
                    # nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            if h_dim == 512:
                modules.append(
                    nn.TransformerEncoderLayer(
                        h_dim, nhead, dim_feedforward, dropout))

            h_in = h_dim
        ##transformer
        # modules.append(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout))
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()
        # print(hidden_dims)
        # exit()
        for h_dim in hidden_dims:

            modules.append(
                nn.Sequential(
                    nn.Linear(h_in, h_dim),
                    # nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            if h_dim == 64:
                modules.append(
                    nn.TransformerDecoderLayer(
                        h_dim, nhead, dim_feedforward, dropout))

            h_in = h_dim
        ##transformer
        # modules.append(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], n_genes),
            # nn.Sigmoid()
        )

        # Build predic
        modules = []
        self.canshu = int(latent_dim / 8)
        self.predic_input = nn.Linear(self.canshu  ,hidden_dims_pre[0])

        # hidden_dims.reverse()

        for i in range(len(hidden_dims_pre) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims_pre[i], hidden_dims_pre[i + 1]),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU())
            )

        self.predictor = nn.Sequential(*modules)
        self.final_pre = nn.Sequential(
            # nn.LayerNorm(hidden_dims_pre[-1]),
            nn.Linear(hidden_dims_pre[-1], n_labels),
            nn.LayerNorm(n_labels),
            # nn.Sigmoid()
            #### crosseloss 不需要sigmoid
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def predic(self, z: Tensor) -> Tensor:
        z_causal   = torch.split(z, self.canshu , dim=1)[0]

        pre_result = self.predic_input(z_causal)
        pre_result = self.predictor(pre_result)
        pre_result = self.final_pre(pre_result)

        return pre_result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        type = self.predic(z)
        return [output, type, input, mu, log_var]
        # return [self.decode(z)]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:

        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
#
# vae=VAE()
# print(vae.decode)

# print(vae)