from __future__ import annotations

from math import ceil, prod

import torch
from torch import nn, pi
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from vector_quantize_pytorch import (
    LFQ
)

from x_transformers import Encoder

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack

from tqdm import tqdm

# ein notation
# b - batch
# c - channels
# h - height
# w - width
# n - raw bits sequence length
# ng - sequence of bit groups

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

def divisible_by(num, den):
    return (num % den) == 0

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

# resnet block

class Block(Module):
    def __init__(
        self,
        dim,
        dropout = 0.
    ):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 3, padding = 1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        *,
        dropout = 0.
    ):
        super().__init__()
        self.block1 = Block(dim, dropout = dropout)
        self.block2 = Block(dim)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + x

# down and upsample

class Upsample(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(
    dim,
    dim_out = None
):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )

# binary quantization vae

class BQVAE(Module):

    @beartype
    def __init__(
        self,
        dim,
        *,
        image_size,
        channels = 3,
        depth = 2,
        proj_in_kernel_size = 7,
        entropy_loss_weight = 0.1,
        lfq_kwargs: dict = dict()
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels

        self.proj_in = nn.Conv2d(channels, dim, proj_in_kernel_size, padding = proj_in_kernel_size // 2)

        self.encoder = ModuleList([])

        # encoder

        curr_dim = dim
        for _ in range(depth):
            self.encoder.append(ModuleList([
                ResnetBlock(curr_dim),
                Downsample(curr_dim, curr_dim * 2)
            ]))

            curr_dim *= 2
            image_size //= 2

        # middle

        self.mid_block = ResnetBlock(curr_dim)

        # codebook

        self.codebook_input_shape = (curr_dim, image_size, image_size)

        # precompute how many bits a single sample is compressed to
        # so maskbit can take this value during sampling

        self.bits_per_image = prod(self.codebook_input_shape)

        self.lfq = LFQ(
            codebook_size = 2, # number of codes is not applicable, as they simply group all the bits and project into tokens for the transformer
            dim = curr_dim,
            **lfq_kwargs
        )

        # decoder

        self.decoder = ModuleList([])

        for _ in range(depth):
            self.decoder.append(ModuleList([
                Upsample(curr_dim, curr_dim // 2),
                ResnetBlock(curr_dim // 2),
            ]))

            curr_dim //= 2

        self.proj_out = nn.Conv2d(curr_dim, channels, 3, padding = 1)

        # aux loss

        self.entropy_loss_weight = entropy_loss_weight

        # tensor typing related

        self._c = channels

    def decode_bits_to_images(
        self,
        bits: Float['b d h w'] | Float['b n'] | Bool['b d h w'] | Bool['b n']
    ):

        if bits.dtype == torch.bool:
            bits = bits.float() * 2 - 1

        if bits.ndim == 2:
            fmap_height, fmap_width = self.codebook_input_shape[-2:]
            bits = rearrange(bits, 'b (d h w) -> b d h w', h = fmap_height, w = fmap_width)

        x = bits

        for upsample, resnet in self.decoder:
            x = upsample(x)
            x = resnet(x)

        recon = self.proj_out(x)

        return recon

    def forward(
        self,
        images: Float['b {self._c} h w'],
        *,
        return_loss = True,
        return_loss_breakdown = False,
        return_quantized_bits = False,
        return_bits_as_bool = False
    ):
        batch = images.shape[0]

        assert images.shape[-2:] == ((self.image_size,) * 2)
        assert not (return_loss and return_quantized_bits)

        x = self.proj_in(images)

        for resnet, downsample in self.encoder:
            x = resnet(x)
            x = downsample(x)

        x = self.mid_block(x)

        bits, _, entropy_aux_loss = self.lfq(x)

        if return_quantized_bits:
            if return_bits_as_bool:
                bits = bits > 0.

            return bits

        assert (bits.numel() // batch) == self.bits_per_image

        x = bits

        for upsample, resnet in self.decoder:
            x = upsample(x)
            x = resnet(x)

        recon = self.proj_out(x)

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
        bits_groups = 2,
        dim_head = 64,
        heads = 8,
        encoder_kwargs: dict = dict(),
        loss_ignore_index = -1,
        train_frac_bits_flipped = 0.05
    ):
        super().__init__()

        vae.eval()
        self.vae = vae

        self.bits_groups = bits_groups
        # bits_group_size (bits per "token") / bit_groups consecutive bits are masked at a time

        assert divisible_by(bits_group_size, bits_groups)
        self.consecutive_bits_to_mask = bits_group_size // bits_groups

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

        self.train_frac_bits_flipped = train_frac_bits_flipped

        # tensor typing

        self._c = vae.channels

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        num_demasking_steps = 18,
        temperature = 1.,
        return_bits = False,
        return_bits_as_bool = False,
    ):
        device = self.device

        seq_len = self.vae.bits_per_image

        bits = torch.zeros(batch_size, seq_len, device = device) # start off all masked, 0.

        # times go from 0. to 1. for `num_demasking_steps`

        times = torch.linspace(0., 1., num_demasking_steps, device = device)
        noise_levels = torch.cos(times * pi * 0.5)
        num_bits_to_mask = (noise_levels * seq_len).long().ceil().clamp(min = 1)

        # iteratively denoise with attention

        for ind, bits_to_mask in tqdm(enumerate(num_bits_to_mask)):
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

        images = self.vae.decode_bits_to_images(bits)

        if not return_bits:
            return images

        if return_bits_as_bool:
            bits = bits > 0.

        return images, bits

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

        num_bits, orig_bits = bits.shape[-1], bits

        # flip a few of the bits, so that the model learns to predict for tokens that are not masked

        if self.train_frac_bits_flipped > 0.:
            num_bits_to_flip = num_bits * self.train_frac_bits_flipped
            flip_mask = torch.rand_like(bits).argsort(dim = -1) < num_bits_to_flip

            bits = torch.where(flip_mask, bits * -1, bits)

        # get the masking fraction, which is a function of time and the noising schedule (we will go with the successful cosine schedule here from Nichol et al)

        times = torch.rand(batch, device = device)
        noise_level = torch.cos(times * pi * 0.5)

        # determine num bit groups and reshape

        assert divisible_by(num_bits, self.consecutive_bits_to_mask)

        bits = rearrange(bits, 'b (ng g) -> b ng g', g = self.consecutive_bits_to_mask)

        bit_group_seq_len = bits.shape[1]

        num_bit_group_mask = (bit_group_seq_len * noise_level).ceil().clamp(min = 1)

        # mask some fraction of the bits

        mask = torch.rand((batch, bit_group_seq_len), device = device).argsort(dim = -1) < num_bit_group_mask

        masked_bits = einx.where('b ng, , b ng g -> b (ng g)', mask, 0., bits) # main contribution of the paper is just this line of code where they mask bits to 0.

        # attention

        preds = self.demasking_transformer(masked_bits)

        # loss mask

        mask = repeat(mask, 'b ng -> b (ng g)', g = self.consecutive_bits_to_mask)

        loss_mask = mask | flip_mask

        # get loss

        labels = (orig_bits[loss_mask] > 0.).long()

        loss = F.cross_entropy(
            preds[loss_mask],
            labels,
            ignore_index = self.loss_ignore_index
        )

        return loss
