from typing import Callable, Iterable
import copy
import torch
import torch.nn as nn

# Disable cuDNN SDPA backend to prevent NaN gradients on H100 (sm_90).
torch.backends.cuda.enable_cudnn_sdp(False)

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.cli import LightningArgumentParser
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

from cod.utils.patch_bugs import *
from cod.models.autoencoder import fp2uint8, PixelConditioner
from cod.utils.no_grad import no_grad, filter_nograd_tensors
from cod.utils.model_loader import load_pretrained_state_dict
from cod.main import DataModule, ReWriteRootSaveConfigCallback, BaseReWriteRootDirCli
from finetuned_one_step_codec.models.models import Discriminator

import logging
logger = logging.getLogger("lightning.pytorch")

OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


# ============================================================
#  LightningModel
# ============================================================

class LightningModel(pl.LightningModule):
    def __init__(self,
                 net: nn.Module,
                 vae: nn.Module,
                 conditioner: PixelConditioner,
                 dmd_denoiser: nn.Module,
                 dmd_trainer: nn.Module,
                 disc_weight_path: str,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 eval_original_model: bool = False,
                 latent: bool = False,
                 dmd_step=10,
                 dmd_learning_rate=1e-5,
                 grad_accum_steps=1,
                 ):
        super().__init__()
        self.vae = vae
        self.conditioner = conditioner
        self.net = net
        self.dmd_trainer = dmd_trainer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.eval_original_model = eval_original_model
        self.latent = latent

        self._strict_loading = False

        self.grad_accum_steps = grad_accum_steps

        self.dmd_learning_rate = dmd_learning_rate

        # dmd
        self.real_denoiser = dmd_denoiser
        self.fake_denoiser = copy.deepcopy(dmd_denoiser)
        self.dmd_step = dmd_step
        self.automatic_optimization = False

        # gan
        self.disc = Discriminator(weight_path=disc_weight_path)

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()

        # disable grad for conditioner and vae
        no_grad(self.conditioner)
        no_grad(self.vae)
        no_grad(self.real_denoiser)
        no_grad(self.fake_denoiser.y_embedder)

        # disable grad for codec encoder and bottleneck
        if self.net.fix_encoder:
            no_grad(self.net.y_embedder.encoder)
            no_grad(self.net.y_embedder.bottleneck)

        # torch.compile
        # self.net.compile()
        self.real_denoiser.compile()
        self.fake_denoiser.compile()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.net.parameters())
        params_trainer = filter_nograd_tensors(self.dmd_trainer.parameters())
        disc = filter_nograd_tensors(self.disc.parameters())

        # dmd params
        params_fake_denoiser = filter_nograd_tensors(self.fake_denoiser.parameters())
        opt_dmd = self.optimizer([
            {"params": params_fake_denoiser, "lr": self.dmd_learning_rate},
            {"params": disc, "lr": self.dmd_learning_rate},
        ])

        # main params
        param_groups = [
            {"params": params_denoiser, },
            {"params": params_trainer,},
        ]
        opt_full = self.optimizer(param_groups)

        if self.lr_scheduler is None:
            return [opt_dmd, opt_full]
        else:
            lr_scheduler = self.lr_scheduler(opt_full)
            return (
                {
                    "optimizer": opt_dmd,
                },
                {
                    "optimizer": opt_full,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "interval": "step",
                        "frequency": 1,
                    },
                },
            )

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        with torch.no_grad():
            if self.latent:
                x = self.vae.encode(y * 2 - 1)
            condition, uncondition = self.conditioner(y)

        # Note : do not separete opt since we use deepspeed zero
        # 1. Forward Pass
        is_dmd_step = (self.trainer.global_step + 1) % self.dmd_step == 0
        out = self.dmd_trainer(
            self.net, self.real_denoiser, self.fake_denoiser, self.disc,
            x, condition, uncondition, metadata, dmd_step = not is_dmd_step
        )
        if is_dmd_step:
            loss = out["loss"]
        else:
            loss = out["fake_loss"]
        self.log("lr", self.optimizers()[1].param_groups[0]['lr'], prog_bar=True, on_step=True, sync_dist=False)
        # self.log("lr_dmd", self.optimizers()[0].param_groups[0]['lr'], prog_bar=True, on_step=True, sync_dist=False)
        self.log_dict(out, on_step=True, prog_bar=True, sync_dist=False)

        # 2. Backward Pass
        self.manual_backward(loss / self.grad_accum_steps)

        # 3. Optimizer Step & Zero Grad (Conditioned on accumulation)
        if (batch_idx + 1) % self.grad_accum_steps == 0:
            sch = self.lr_schedulers()
            opt_dmd, opt_full = self.optimizers()
            if is_dmd_step:
                opt_full.step()
                opt_full.zero_grad()
            else:
                opt_dmd.step()
                opt_dmd.zero_grad()
            if sch is not None:
                sch.step()

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch

        # model.inference : input y in [0, 1], output recon in [-1, 1]
        samples = self.net.inference(y)

        # fp32 -1,1 -> uint8 0,255
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx)
        return samples

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.net.state_dict(
            destination=destination,
            prefix=prefix+"net.",
            keep_vars=keep_vars)
        self.real_denoiser.state_dict(
            destination=destination,
            prefix=prefix+"real_denoiser.",
            keep_vars=keep_vars)
        self.fake_denoiser.state_dict(
            destination=destination,
            prefix=prefix+"fake_denoiser.",
            keep_vars=keep_vars)
        self.dmd_trainer.state_dict(
            destination=destination,
            prefix=prefix+"dmd_trainer.",
            keep_vars=keep_vars)
        self.disc.state_dict(
            destination=destination,
            prefix=prefix+"disc.",
            keep_vars=keep_vars)
        return destination


# ============================================================
#  CLI
# ============================================================

class ReWriteRootDirCli(BaseReWriteRootDirCli):

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--dmd_ckpt_path", type=str, default=None, help="Path to pretrained checkpoint")

    def _load_pretrained(self):
        dmd_ckpt_path = self._get(self.config, "dmd_ckpt_path")
        assert dmd_ckpt_path is not None
        print(f"Loading DMD weights from {dmd_ckpt_path}")

        pretrained_dmd_dict = load_pretrained_state_dict(dmd_ckpt_path)

        dmd_dict = {}
        for k, v in pretrained_dmd_dict.items():
            if k.startswith("ema_denoiser."):
                dmd_dict[k.replace("ema_denoiser.", "")] = v

        self.model.real_denoiser.load_state_dict(dmd_dict, strict=True)
        self.model.fake_denoiser.load_state_dict(dmd_dict, strict=True)


if __name__ == "__main__":

    cli = ReWriteRootDirCli(LightningModel, DataModule,
                            auto_configure_optimizers=False,
                            save_config_callback=ReWriteRootSaveConfigCallback,
                            save_config_kwargs={"overwrite": True})
