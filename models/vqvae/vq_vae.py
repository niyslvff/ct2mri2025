import torch
from torch import nn
from torch.nn import functional as F
try:
    from base import BaseVAE
except:
    from .base import BaseVAE
from typing import List, Callable, Union, Any, TypeVar, Tuple

Tensor = TypeVar('torch.tensor')

class _nn:
    @staticmethod
    def conv_nd(dims, *args, **kwargs):
        if dims == 1:
            return nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            return nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            return nn.Conv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")

    @staticmethod
    def convtranspose_nd(dims, *args, **kwargs):
        if dims == 1:
            return nn.ConvTranspose1d(*args, **kwargs)
        elif dims == 2:
            return nn.ConvTranspose2d(*args, **kwargs)
        elif dims == 3:
            return nn.ConvTranspose3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")

    @staticmethod
    def batch_norm_nd(dims, *args, **kwargs):
        if dims == 2:
            return nn.BatchNorm2d(*args, **kwargs)
        elif dims == 3:
            return nn.BatchNorm3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")

    @staticmethod
    def avg_pool_nd(dims, *args, **kwargs):
        if dims == 1:
            return nn.AvgPool1d(*args, **kwargs)
        elif dims == 2:
            return nn.AvgPool2d(*args, **kwargs)
        elif dims == 3:
            return nn.AvgPool3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")



class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self,
                 out_channels: int,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25,
                 dims: int = 2,
        ):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.dims = dims

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        if self.dims == 2:
            latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        elif self.dims == 3:
            latents = latents.permute(0, 2, 3, 4, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        if self.dims == 2:
            quantized_latents, vq_loss = quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]
        elif self.dims == 3:
            quantized_latents, vq_loss = quantized_latents.permute(0, 4, 1, 2, 3).contiguous(), vq_loss

        return quantized_latents, vq_loss


class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dims=2):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(_nn.conv_nd(dims, in_channels, out_channels,
                                                  kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      _nn.conv_nd(dims, out_channels, out_channels,
                                                  kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 dims: int = 2,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    _nn.conv_nd(dims, in_channels, out_channels=h_dim,
                                kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                _nn.conv_nd(dims, in_channels, in_channels,
                            kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels, dims))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                _nn.conv_nd(dims, in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(
            out_channels,
            num_embeddings,
            embedding_dim,
            self.beta,
            dims,
        )

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                _nn.conv_nd(
                    dims,
                    embedding_dim,
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1], dims))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    _nn.convtranspose_nd(dims, hidden_dims[i],
                                         hidden_dims[i + 1],
                                         kernel_size=4,
                                         stride=2,
                                         padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                _nn.convtranspose_nd(dims, hidden_dims[-1],
                                     out_channels=out_channels,
                                     kernel_size=4,
                                     stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # print('encode_input_shape', result.shape)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        # print('decode_input_shape', z.shape)
        result = self.decoder(z)
        # print('decoded_shape', result.shape)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss': vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
