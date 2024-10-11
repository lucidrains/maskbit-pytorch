from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from vector_quantize_pytorch import (
    LFQ
)

from x_transformers import (
    Encoder,
    NonAutoregressiveWrapper
)

from einops.layers.torch import Rearrange
from einops import rearrange, pack, unpack

# ein notation
# b - batch
# c - channels
# h - height
# w - width
# n - raw bits sequence length

# tensor typing

import jaxtyping
from jaxtyping import jaxtyped
from beartype import beartype
from beartype.door import is_bearable

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    t, packed_shape = pack([t], pattern)

    def inverse(t, unpack_pattern = None):
        unpack_pattern = default(unpack_pattern, pattern)
        return unpack(t, packed_shape, unpack_pattern)[0]

    return t, inverse

# binary quantization vae

class BQVAE(Module):

    @beartype
    def __init__(
        self,
        dim,
        *,
        channels = 3,
        entropy_loss_weight = 0.1,
        lfq_kwargs: dict = dict()
    ):
        super().__init__()
        self.channels = channels

        self.encoder = nn.Conv2d(channels, dim, 1)

        self.lfq = LFQ(
            codebook_size = 2, # number of codes is not applicable, as they simply group all the bits and project into tokens for the transformer
            dim = dim,
            **lfq_kwargs
        )

        self.entropy_loss_weight = entropy_loss_weight

        self.decoder = nn.Conv2d(dim, channels, 1)

        # tensor typing related

        self._c = channels

    def forward(
        self,
        images: Float['b {self._c} h w'],
        *,
        return_loss = False,
        return_loss_breakdown = False,
        return_quantized_bits = False
    ):
        assert not (return_loss and return_quantized_bits)

        x = self.encoder(images)

        quantized, _, entropy_aux_loss = self.lfq(x)

        if return_quantized_bits:
            return quantized

        recon = self.decoder(x)

        if not return_loss:
            return recon

        recon_loss = F.mse_loss(images, recon)

        total_loss = (
            recon_loss +
            self.entropy_loss_weight * entropy_aux_loss
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (recon_loss, entropy_aux_loss)

# class

class MaskBit(Module):

    @beartype
    def __init__(
        self,
        vae: BQVAE,
        *,
        bits_group_size,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        encoder_kwargs: dict = dict(),
        loss_ignore_index = -1
    ):
        super().__init__()

        vae.eval()
        self.vae = vae

        self.to_tokens = nn.Sequential(
            Rearrange('b (n g) -> b n g', g = bits_group_size),
            nn.Linear(bits_group_size, dim)
        )

        self.transformer = Encoder(
            dim = dim,
            depth = depth,
            attn_dim_head = dim_head,
            heads = heads,
            **encoder_kwargs
        )

        self.to_unmasked_bit_pred = nn.Sequential(
            nn.Linear(dim, bits_group_size * 2),
            Rearrange('... (g bits) -> ... g bits', bits = 2)
        )

        self.loss_ignore_index = loss_ignore_index

        # tensor typing

        self._c = vae.channels

    def forward(
        self,
        images: Float['b {self._c} h w']
    ):
        bits = self.vae(
            images,
            return_quantized_bits = True
        )

        bit_seq, unpack_one = pack_one(bits, 'b *')

        tokens = self.to_tokens(bit_seq)

        tokens = self.transformer(tokens)

        pred = self.to_unmasked_bit_pred(tokens)

        return pred
