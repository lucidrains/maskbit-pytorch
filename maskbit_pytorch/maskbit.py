from __future__ import annotations

from math import ceil

import torch
from torch import nn, pi
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from vector_quantize_pytorch import (
    LFQ
)

from x_transformers import Encoder

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

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def calc_entropy(logits):
    prob = logits.softmax(dim = -1)
    return (-prob * log(prob)).sum(dim = -1)

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

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
        return_loss = True,
        return_loss_breakdown = False,
        return_quantized_bits = False,
        return_bits_as_bool = False
    ):
        assert not (return_loss and return_quantized_bits)

        x = self.encoder(images)

        quantized, _, entropy_aux_loss = self.lfq(x)

        if return_quantized_bits:
            if return_bits_as_bool:
                quantized = quantized > 0.

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

        self.demasking_transformer = nn.Sequential(
            Rearrange('b (n g) -> b n g', g = bits_group_size),
            nn.Linear(bits_group_size, dim),
            Encoder(
                dim = dim,
                depth = depth,
                attn_dim_head = dim_head,
                heads = heads,
                **encoder_kwargs
            ),
            nn.Linear(dim, bits_group_size * 2),
            Rearrange('b n (g bits) -> b (n g) bits', bits = 2)
        )

        self.loss_ignore_index = loss_ignore_index

        # tensor typing

        self._c = vae.channels

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        seq_len,
        batch_size = 1,
        num_demasking_steps = 18,
        temperature = 1.,
        return_bits_as_bool = False
    ):
        device = self.device

        bits = torch.zeros(batch_size, seq_len, device = device) # start off all masked, 0.

        # times go from 0. to 1. for `num_demasking_steps`

        times = torch.linspace(0., 1., num_demasking_steps, device = device)
        noise_levels = torch.cos(times * pi * 0.5)
        num_bits_to_mask = (noise_levels * seq_len).long().ceil().clamp(min = 1)

        # iteratively denoise with attention

        for ind, bits_to_mask in enumerate(num_bits_to_mask):
            is_first = ind == 0

            # if not the first step, mask by the previous step's bit predictions with highest entropy

            if not is_first:
                entropy = calc_entropy(logits)
                remask_indices = entropy.topk(bits_to_mask.item(), dim = -1).indices
                bits.scatter_(1, remask_indices, 0.) # recall they use 0. for masking

            # ask the attention network to predict the bits

            logits = self.demasking_transformer(bits)

            # sample the bits

            bits = gumbel_sample(logits, temperature = temperature)
            bits = (bits * 2 - 1.) # bits are -1. or +1

        if return_bits_as_bool:
            bits = bits > 0.

        return bits

    def forward(
        self,
        images: Float['b {self._c} h w']
    ):
        batch, device = images.shape[0], self.device

        with torch.no_grad():
            self.vae.eval()

            bits = self.vae(
                images,
                return_loss = False,
                return_quantized_bits = True
            )

        # pack the bits into one long sequence

        bits, _ = pack_one(bits, 'b *')

        num_bits = bits.shape[-1]

        # get the masking fraction, which is a function of time and the noising schedule (we will go with the successful cosine schedule here from Nichol et al)

        times = torch.rand(batch, device = device)
        noise_level = torch.cos(times * pi * 0.5)
        num_bits_mask = (num_bits * noise_level).ceil().clamp(min = 1)

        # mask some fraction of the bits

        mask = torch.rand_like(bits).argsort(dim = -1) < num_bits_mask
        bits.masked_fill_(mask, 0.) # main contribution of the paper is just this line of code where they mask bits to 0.

        # attention

        preds = self.demasking_transformer(bits)

        # get loss

        labels = (bits[mask] == 1.).long()

        loss = F.cross_entropy(
            preds[mask],
            labels,
            ignore_index = self.loss_ignore_index
        )

        return loss
