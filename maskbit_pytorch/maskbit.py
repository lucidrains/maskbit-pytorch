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
        lfq_kwargs: dict = dict()
    ):
        super().__init__()

        self.lfq = LFQ(
            codebook_size = 2, # number of codes is not applicable, as they simply group all the bits and project into tokens for the transformer
            dim = dim,
            **lfq_kwargs
        )

    def forward(
        self,
        images: Float['b c h w']
    ):
        return images.sum()

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
        encoder_kwargs: dict = dict()
    ):
        super().__init__()
        self.to_tokens = nn.Linear(bits_group_size, dim)

        self.transformer = Encoder(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            **encoder_kwargs
        )

        self.to_unmasked_bit_pred = nn.Linear(dim, bits_group_size)

    def forward(
        self
    ):
        return
