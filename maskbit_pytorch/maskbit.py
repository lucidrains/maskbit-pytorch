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

# binary quantization vae

class BQVAE(Module):
    def __init__(
        self,
        dim,
        channels = 3,
        lfq_kwargs: dict = dict()
    ):
        super().__init__()
        self._c = channels
        self.encoder = nn.Conv2d(3, dim, 1)

        self.lfq = LFQ(
            codebook_size = 2, # number of codes is not applicable, as they simply group all the bits and project into tokens for the transformer
            dim = dim,
            **lfq_kwargs
        )

        self.decoder = nn.Conv2d(dim, 3, 1)

    def forward(
        self,
        images: Float['b {self._c} h w'],
        return_loss = False
    ):
        x = self.encoder(images)

        quantized, *_ = self.lfq(x)

        recon = self.decoder(x)

        if not return_loss:
            return recon

        return F.mse_loss(images, recon)

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

        self.to_tokens = nn.Linear(bits_group_size, dim)

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

    def forward(
        self,
        bits: Bool['b n'],
        bit_mask: Bool['b n']  # for iterative masking for NAR decoding
    ):
        return
