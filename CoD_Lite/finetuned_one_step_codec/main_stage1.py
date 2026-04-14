from typing import Callable, Iterable, Sequence
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from peft import LoraConfig, get_peft_model
import lpips

from cod.utils.patch_bugs import *
from cod.models.autoencoder import fp2uint8, PixelConditioner
from cod.utils.no_grad import no_grad, filter_nograd_tensors
from cod.utils.model_loader import load_pretrained_model, load_pretrained_state_dict
from cod.main import DataModule, ReWriteRootSaveConfigCallback, BaseReWriteRootDirCli
from finetuned_one_step_codec.models.models import OneStepCoD

import logging
logger = logging.getLogger("lightning.pytorch")

OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


# ============================================================
#  FinetunedOneStepCoD
# ============================================================

class FinetunedOneStepCoD(OneStepCoD):
    def __init__(self, noise_level=0.0, fix_encoder=False, pretrain_prefix="net.", *args, **kwargs):
        self.pretrain_prefix = pretrain_prefix
        super().__init__(noise_level=noise_level, fix_encoder=fix_encoder, *args, **kwargs)

    def load_pretrained(self, net_ckpt_path, pretrained_ema=True):
        assert net_ckpt_path is not None
        print(f"Loading Network weights from {net_ckpt_path} (ema=True, strict=True)")
        pretrained_dict = load_pretrained_state_dict(net_ckpt_path)
        model_dict = self.state_dict()

        matched_dict = {}
        skipped_keys = []
        for k, v in pretrained_dict.items():
            if not k.startswith(self.pretrain_prefix):
                continue
            new_k = k[len(self.pretrain_prefix):]
            if new_k in model_dict:
                if model_dict[new_k].shape == v.shape:
                    matched_dict[new_k] = v
                else:
                    skipped_keys.append((k, f"shape mismatch: {v.shape} vs {model_dict[new_k].shape}"))
            else:
                skipped_keys.append((k, "key not found"))

        model_dict.update(matched_dict)
        self.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(matched_dict)} parameters, skipped {len(skipped_keys)}")


# ============================================================
#  FinetunedOneStepCoDTrainer
# ============================================================

class FinetunedOneStepCoDTrainer(nn.Module):
    def __init__(
            self,
            net_loss: nn.Module = None,
            net_loss_weight=1.0,
            lpips_vgg_weight=1.0,
            lpips_alex_weight=0.5,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.net_loss = net_loss
        self.net_loss_weight = net_loss_weight
        self.lpips_vgg_weight = lpips_vgg_weight
        self.lpips_alex_weight = lpips_alex_weight

        self.lpips_fn_vgg = lpips.LPIPS(net='vgg').eval()
        self.lpips_fn_alex = lpips.LPIPS(net='alex').eval()
        no_grad(self.lpips_fn_vgg)
        no_grad(self.lpips_fn_alex)

    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def __call__(self, net, x, condition, uncondition, metadata=None):
        net_recon, net_loss_dict = self.net_loss(condition, net)

        l1_loss = (net_recon - x).abs()
        lpips_loss = self.lpips_vgg_weight * self.lpips_fn_vgg(net_recon, x) + \
                     self.lpips_alex_weight * self.lpips_fn_alex(net_recon, x)

        net_loss = net_loss_dict['net_loss']
        del net_loss_dict['net_loss']

        out = net_loss_dict
        out.update(
            l1=l1_loss.mean(),
            lpips=lpips_loss.mean(),
            loss=l1_loss.mean() + lpips_loss.mean() +
                 self.net_loss_weight * net_loss.mean()
        )
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self.net_loss.state_dict(
            destination=destination,
            prefix=prefix + "net_loss.",
            keep_vars=keep_vars)
        return destination


# ============================================================
#  LoRA Utilities
# ============================================================

def find_lora_target_modules(
    model: nn.Module,
    target_types: Sequence[nn.Module] = (nn.Linear, nn.Conv2d, nn.Conv1d),
    exclude_modules: Sequence[str] = ("y_embedder",)
) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        if any(ex in name for ex in exclude_modules):
            continue
        if isinstance(module, target_types):
            # Skip grouped Conv2d (LoRA doesn't support rank not divisible by groups)
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                continue
            target_modules.append(name)
    return target_modules


# ============================================================
#  LightningModel
# ============================================================

class LightningModel(pl.LightningModule):
    def __init__(self,
                 net: nn.Module,
                 vae: nn.Module,
                 conditioner: PixelConditioner,
                 codec_trainer: nn.Module,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 eval_original_model: bool = False,
                 latent: bool = False,
                 use_lora: bool = True,
                 ):
        super().__init__()
        self.vae = vae
        self.conditioner = conditioner
        self.net = net
        self.codec_trainer = codec_trainer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_original_model = eval_original_model
        self.latent = latent
        self._strict_loading = False

        if use_lora:
            lora_target_modules = find_lora_target_modules(
                self.net,
                target_types=(nn.Linear, nn.Conv2d, nn.Conv1d),
                exclude_modules=["y_embedder", "conv2"]
            )
            lora_config = LoraConfig(
                r=32, lora_alpha=32,
                target_modules=lora_target_modules,
                lora_dropout=0.0, bias="none",
                modules_to_save=["y_embedder", "conv2"]
            )
            self.net = get_peft_model(self.net, lora_config)
            print("PEFT Model Trainable Parameters:")
            self.net.print_trainable_parameters()
        else:
            print("Full fine-tuning (no LoRA)")

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        no_grad(self.conditioner)
        no_grad(self.vae)
        self.net.compile()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_net = filter_nograd_tensors(self.net.parameters())
        params_trainer = filter_nograd_tensors(self.codec_trainer.parameters())
        param_groups = [
            {"params": params_net},
            {"params": params_trainer},
        ]
        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        if self.lr_scheduler is None:
            return dict(optimizer=optimizer)
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        with torch.no_grad():
            if self.latent:
                x = self.vae.encode(y * 2 - 1)
            condition, uncondition = self.conditioner(y)
        loss = self.codec_trainer(self.net, x, condition, uncondition, metadata)
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch
        samples = self.net.inference(y)
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx):
        return self.predict_step(batch, batch_idx)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.net.state_dict(destination=destination, prefix=prefix + "net.", keep_vars=keep_vars)
        self.codec_trainer.state_dict(destination=destination, prefix=prefix + "codec_trainer.", keep_vars=keep_vars)
        return destination


# ============================================================
#  CLI
# ============================================================

class ReWriteRootDirCli(BaseReWriteRootDirCli):

    def _load_pretrained(self):
        pretrained_ckpt_path = self._get(self.config, "pretrained_ckpt_path")
        if pretrained_ckpt_path is not None:
            pretrained_ema = self._get(self.config, "pretrained_ema")
            self.model = load_pretrained_model(self.model, pretrained_ckpt_path, pretrained_ema, strict=False, log=True)


if __name__ == "__main__":

    cli = ReWriteRootDirCli(LightningModel, DataModule,
                            auto_configure_optimizers=False,
                            save_config_callback=ReWriteRootSaveConfigCallback,
                            save_config_kwargs={"overwrite": True})
