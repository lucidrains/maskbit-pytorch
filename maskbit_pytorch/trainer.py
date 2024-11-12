from pathlib import Path
from shutil import rmtree
from functools import partial

from beartype import beartype

import torch
from torch import nn, tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader, random_split

from adam_atan2_pytorch import Adam

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from maskbit_pytorch.maskbit import BQVAE, MaskBit

from einops import rearrange

from accelerate import (
    Accelerator,
    DistributedType,
    DistributedDataParallelKwargs
)

from ema_pytorch import EMA

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# helper functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def noop(*args, **kwargs):
    pass

def find_index(arr, cond):
    for ind, el in enumerate(arr):
        if cond(el):
            return ind
    return None

def find_and_pop(arr, cond, default = None):
    ind = find_index(arr, cond)

    if exists(ind):
        return arr.pop(ind)

    if callable(default):
        return default()

    return default

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# image related helpers fnuctions and dataset

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')
        assert len(self) > 0

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# vae trainer class

class BQVAETrainer(Module):

    @beartype
    def __init__(
        self,
        vae: BQVAE,
        *,
        folder,
        num_train_steps,
        batch_size,
        image_size,
        lr = 3e-4,
        grad_accum_every = 1,
        max_grad_norm = None,
        discr_max_grad_norm = None,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        use_ema = True,
        ema_beta = 0.995,
        ema_update_after_step = 0,
        ema_update_every = 1,
        ema_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()

        # instantiate accelerator

        kwargs_handlers = accelerate_kwargs.get('kwargs_handlers', [])

        ddp_kwargs = find_and_pop(
            kwargs_handlers,
            lambda x: isinstance(x, DistributedDataParallelKwargs),
            partial(DistributedDataParallelKwargs, find_unused_parameters = True)
        )

        ddp_kwargs.find_unused_parameters = True
        kwargs_handlers.append(ddp_kwargs)
        accelerate_kwargs.update(kwargs_handlers = kwargs_handlers)

        self.accelerator = Accelerator(**accelerate_kwargs)

        # vae

        self.vae = vae

        # training params

        self.register_buffer('steps', tensor(0))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        all_parameters = set(vae.parameters())
        discr_parameters = set(vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters

        self.vae_parameters = vae_parameters

        # optimizers

        self.optim = Adam(vae_parameters, lr = lr)
        self.discr_optim = Adam(discr_parameters, lr = lr)

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # create dataset

        self.ds = ImageDataset(folder, image_size)

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True
        )

        # prepare with accelerator

        (
            self.vae,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.vae,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl
        )

        self.use_ema = use_ema

        if use_ema:
            self.ema_vae = EMA(vae, update_after_step = ema_update_after_step, update_every = ema_update_every, **ema_kwargs)
            self.ema_vae = self.accelerator.prepare(self.ema_vae)

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model = self.accelerator.get_state_dict(self.vae),
            optim = self.optim.state_dict(),
            discr_optim = self.discr_optim.state_dict()
        )

        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        vae = self.accelerator.unwrap_model(self.vae)
        vae.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])
        self.discr_optim.load_state_dict(pkg['discr_optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        acc = self.accelerator
        device = self.device

        steps = int(self.steps.item())

        self.vae.train()
        discr = self.vae.module.discr if self.is_distributed else self.vae.discr

        if self.use_ema:
            ema_vae = self.ema_vae.module if self.is_distributed else self.ema_vae

        # logs

        logs = dict()

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            img = next(self.dl_iter)
            img = img.to(device)

            with acc.autocast():
                loss = self.vae(
                    img,
                    return_loss = True
                )

            acc.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            acc.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # update discriminator

        if exists(discr):
            self.discr_optim.zero_grad()

            for _ in range(self.grad_accum_every):
                img = next(self.dl_iter)
                img = img.to(device)

                loss = self.vae(
                    img,
                    return_discr_loss = True
                )

                acc.backward(loss / self.grad_accum_every)

                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

            if exists(self.discr_max_grad_norm):
                acc.clip_grad_norm_(discr.parameters(), self.discr_max_grad_norm)

            self.discr_optim.step()

        # log

        self.print(f"{steps}: vae loss: {logs['loss']:.3f} - discr loss: {logs['discr_loss']:.3f}")

        # update exponential moving averaged generator

        if self.use_ema:
            ema_vae.update()

        # sample results every so often

        if not (steps % self.save_results_every):
            vaes_to_evaluate = ((self.vae, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = ((ema_vae.ema_model, f'{steps}.ema'),) + vaes_to_evaluate

            for model, filename in vaes_to_evaluate:
                model.eval()

                valid_data = next(self.valid_dl_iter)
                valid_data = valid_data.to(device)

                _, recons, _ = model(valid_data, return_details = True)

                # else save a grid of images

                imgs_and_recons = rearrange([valid_data, recons], 'r b ... -> (b r) ...')

                imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

                logs['reconstructions'] = grid

                save_image(grid, str(self.results_folder / f'{filename}.png'))

            self.print(f'{steps}: saving to {str(self.results_folder)}')

        # save model every so often

        acc.wait_for_everyone()

        if self.is_main and not (steps % self.save_model_every):
            state_dict = acc.unwrap_model(self.vae).state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            acc.save(state_dict, model_path)

            if self.use_ema:
                ema_state_dict = acc.unwrap_model(self.ema_vae).state_dict()
                model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
                acc.save(ema_state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def forward(self):
        
        while self.steps < self.num_train_steps:
            logs = self.train_step()

        self.print('training complete')

# maskbit trainer

class MaskBitTrainer(Module):
    def __init__(
        self,
        maskbit: MaskBit,
        folder,
        num_train_steps,
        batch_size,
        image_size,
        lr = 3e-4,
        grad_accum_every = 1,
        max_grad_norm = None,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()

        # instantiate accelerator

        kwargs_handlers = accelerate_kwargs.get('kwargs_handlers', [])

        ddp_kwargs = find_and_pop(
            kwargs_handlers,
            lambda x: isinstance(x, DistributedDataParallelKwargs),
            partial(DistributedDataParallelKwargs, find_unused_parameters = True)
        )

        ddp_kwargs.find_unused_parameters = True
        kwargs_handlers.append(ddp_kwargs)
        accelerate_kwargs.update(kwargs_handlers = kwargs_handlers)

        self.accelerator = Accelerator(**accelerate_kwargs)

        # training params

        self.register_buffer('steps', tensor(0))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # model

        self.maskbit = maskbit

        # optimizers

        self.optim = Adam(maskbit.parameters(), lr = lr)

        self.max_grad_norm = max_grad_norm

        # create dataset

        self.ds = ImageDataset(folder, image_size)

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True
        )

        # prepare with accelerator

        (
            self.maskbit,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.maskbit,
            self.optim,
            self.dl,
            self.valid_dl
        )

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model = self.accelerator.get_state_dict(self.maskbit),
            optim = self.optim.state_dict(),
        )

        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        maskbit = self.accelerator.unwrap_model(self.maskbit)
        maskbit.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        acc = self.accelerator
        device = self.device

        steps = int(self.steps.item())

        self.maskbit.train()

        # logs

        logs = dict()

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            img = next(self.dl_iter)
            img = img.to(device)

            with acc.autocast():
                loss = self.maskbit(img)

            acc.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            acc.clip_grad_norm_(self.maskbit.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: maskbit loss: {logs['loss']:.3f}")

        # save model every so often

        acc.wait_for_everyone()

        if self.is_main and not (steps % self.save_model_every):
            state_dict = acc.unwrap_model(self.maskbit).state_dict()
            model_path = str(self.results_folder / f'maskbit.{steps}.pt')
            acc.save(state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def forward(self):

        while self.steps < self.num_train_steps:
            logs = self.train_step()

        self.print('training complete')
