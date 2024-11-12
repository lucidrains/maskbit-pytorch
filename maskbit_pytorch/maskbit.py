from __future__ import annotations

from math import ceil, prod, log2
from functools import cache

import torch
from torch import nn, pi, tensor
import torch.nn.functional as F
import torch.distributed as dist
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

def is_empty(t: Tensor):
    return t.numel() == 0

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

# distributed helpers

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist.all_reduce(t)
    t = t / dist.get_world_size()
    return t

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

# adversarial related

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

class ScalarEMA(Module):
    def __init__(self, decay: float):
        super().__init__()
        self.decay = decay

        self.register_buffer('initted', tensor(False))
        self.register_buffer('ema', tensor(0.))

    @torch.no_grad()
    def forward(
        self,
        values: Float['b']
    ):
        if is_empty(values):
            return

        values = values.mean()
        values = maybe_distributed_mean(values)

        if not self.initted:
            self.ema.copy_(values)
            self.initted.copy_(tensor(True))
            return

        self.ema.lerp_(values, 1. - self.decay)

class ChanRMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = rearrange(self.gamma, 'c -> c 1 1')
        return F.normalize(x, dim = 1) * self.scale * (gamma + 1)

class Discriminator(Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        channels = 3,
        init_kernel_size = 5,
        ema_decay = 0.99,
    ):
        super().__init__()
        first_dim, *_, last_dim = dims
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = ModuleList([])

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(channels, first_dim, init_kernel_size, padding = init_kernel_size // 2),
                nn.SiLU()
            )
        )

        for dim_in, dim_out in dim_pairs:
            layer = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                ChanRMSNorm(dim_out),
                nn.SiLU()
            )

            self.layers.append(layer)

        dim = last_dim

        self.to_logits = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.SiLU(),
            nn.Conv2d(dim, 1, 4)
        )

        # for keeping track of the exponential moving averages of real and fake predictions
        # for the lecam divergence gan technique employed https://arxiv.org/abs/2104.03310

        self.ema_real = ScalarEMA(ema_decay)
        self.ema_fake = ScalarEMA(ema_decay)

    def forward(
        self,
        x: Float['b c h w'],
        is_real: bool | Bool['b'] | None = None
    ):
        batch, device = x.shape[0], x.device

        for layer in self.layers:
            x = layer(x)

        preds = self.to_logits(x)

        if not self.training or not exists(is_real):
            return preds

        if isinstance(is_real, bool):
            is_real = torch.full((batch,), is_real, dtype = torch.bool, device = device)

        is_fake = ~is_real

        preds_real = preds[is_real]
        preds_fake = preds[is_fake]

        self.ema_real(preds_real)
        self.ema_fake(preds_fake)

        reg_loss = 0.

        if not is_empty(preds_real) and self.ema_fake.initted:
            reg_loss = reg_loss + ((preds_real - self.ema_fake.ema) ** 2).mean()

        if not is_empty(preds_fake) and self.ema_real.initted:
            reg_loss = reg_loss + ((preds_fake - self.ema_real.ema) ** 2).mean()

        return preds, reg_loss

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
        reg_loss_weight = 1e-2,
        gen_loss_weight = 1e-1,
        lfq_kwargs: dict = dict(),
        discr_kwargs: dict = dict()
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

        # discriminator

        self.discr = Discriminator(
            dims = (dim,) * int(log2(image_size) - 2),
            channels = channels,
            **discr_kwargs
        )

        # aux loss

        self.entropy_loss_weight = entropy_loss_weight

        self.reg_loss_weight  = reg_loss_weight

        self.gen_loss_weight = gen_loss_weight

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
        return_discr_loss = False,
        return_details = False,
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

        if return_discr_loss:
            images = images.requires_grad_()
            recon = recon.detach()

            discr_real_logits, reg_loss_real = self.discr(images, is_real = True)
            discr_fake_logits, reg_loss_fake = self.discr(recon, is_real = False)

            discr_loss = hinge_discr_loss(discr_fake_logits, discr_real_logits)

            reg_loss = (reg_loss_real + reg_loss_fake) / 2

            loss = discr_loss + reg_loss * self.reg_loss_weight

            if not return_details:
                return loss

            return loss, recon, (discr_loss, reg_loss_real, reg_loss_fake)

        if not return_loss:
            return recon

        recon_loss = F.mse_loss(images, recon)

        discr_fake_logits = self.discr(recon)

        gen_loss = hinge_gen_loss(discr_fake_logits)

        total_loss = (
            recon_loss +
            entropy_aux_loss * self.entropy_loss_weight +
            gen_loss * self.gen_loss_weight
        )

        if not return_details:
            return total_loss

        return total_loss, recon, (recon_loss, entropy_aux_loss, gen_loss)

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

    def parameters(self):
        return self.demasking_transformer.parameters()

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
