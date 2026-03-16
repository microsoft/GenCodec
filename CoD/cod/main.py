import os
import re
import copy
import time
import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Any, Callable, Iterable, Optional, Union, Sequence, Mapping, Dict

from cod.utils.patch_bugs import *
from cod.utils.model_loader import load_pretrained_model
from cod.utils.no_grad import no_grad, filter_nograd_tensors
from cod.utils.copy import copy_params

from cod.models.autoencoder import PixelAE, fp2uint8, PixelConditioner
from cod.callbacks.simple_ema import SimpleEMA

from lightning import Trainer, LightningModule
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset

import logging
logger = logging.getLogger("lightning.pytorch")


# ============================================================
#  Collate functions
# ============================================================

def mirco_batch_collate_fn(batch):
    batch = copy.deepcopy(batch)
    new_batch = []
    for micro_batch in batch:
        new_batch.extend(micro_batch)
    x, y, metadata = list(zip(*new_batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except:
            pass
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return x, y, stacked_metadata

def collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except:
            pass
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return x, y, stacked_metadata

def eval_collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return x, y, metadata


# ============================================================
#  DataModule
# ============================================================

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]

class DataModule(pl.LightningDataModule):
    def __init__(self,
                train_dataset:Dataset=None,
                eval_dataset:Dataset=None,
                pred_dataset:Dataset=None,
                train_batch_size=64,
                train_num_workers=16,
                train_prefetch_factor=8,
                eval_batch_size=32,
                eval_num_workers=4,
                pred_batch_size=32,
                pred_num_workers=4,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.pred_dataset = pred_dataset
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor
        self.eval_batch_size = eval_batch_size
        self.pred_batch_size = pred_batch_size
        self.pred_num_workers = pred_num_workers
        self.eval_num_workers = eval_num_workers
        self._train_dataloader = None

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        micro_batch_size = getattr(self.train_dataset, "micro_batch_size", None)
        if micro_batch_size is not None:
            assert self.train_batch_size % micro_batch_size == 0
            dataloader_batch_size = self.train_batch_size // micro_batch_size
            train_collate_fn = mirco_batch_collate_fn
        else:
            dataloader_batch_size = self.train_batch_size
            train_collate_fn = collate_fn

        if not isinstance(self.train_dataset, IterableDataset):
            sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        else:
            sampler = None

        self._train_dataloader = DataLoader(
            self.train_dataset,
            dataloader_batch_size,
            timeout=6000,
            num_workers=self.train_num_workers,
            prefetch_factor=self.train_prefetch_factor,
            collate_fn=train_collate_fn,
            sampler=sampler,
        )
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(self.eval_dataset, self.eval_batch_size,
                          num_workers=self.eval_num_workers,
                          prefetch_factor=2,
                          sampler=sampler,
                          collate_fn=eval_collate_fn
                )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.pred_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(self.pred_dataset, batch_size=self.pred_batch_size,
                          num_workers=self.pred_num_workers,
                          prefetch_factor=4,
                          sampler=sampler,
                          collate_fn=eval_collate_fn
               )


# ============================================================
#  LightningModel
# ============================================================

class LightningModel(pl.LightningModule):
    def __init__(self,
                 vae: nn.Module,
                 conditioner: PixelConditioner,
                 denoiser: nn.Module,
                 diffusion_trainer: nn.Module,
                 diffusion_sampler: nn.Module,
                 ema_tracker: SimpleEMA=None,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 eval_original_model: bool = False,
                 latent: bool = False,
                 ):
        super().__init__()
        self.vae = vae
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_trainer = diffusion_trainer
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.eval_original_model = eval_original_model
        self.latent = latent

        self._strict_loading = False

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)

        no_grad(self.conditioner)
        no_grad(self.vae)
        no_grad(self.ema_denoiser)

        self.denoiser.compile()
        self.ema_denoiser.compile()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        params_trainer = filter_nograd_tensors(self.diffusion_trainer.parameters())
        params_sampler = filter_nograd_tensors(self.diffusion_sampler.parameters())
        param_groups = [
            {"params": params_denoiser, },
            {"params": params_trainer,},
            {"params": params_sampler, "lr": 1e-3},
        ]
        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        if self.lr_scheduler is None:
            return dict(
                optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )

    def on_validation_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    def on_train_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        with torch.no_grad():
            if self.latent:
                x = self.vae.encode(y * 2 - 1)
            condition, uncondition = self.conditioner(y)
        loss = self.diffusion_trainer(self.denoiser, self.ema_denoiser, self.diffusion_sampler, x, condition, uncondition, metadata)
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch
        with torch.no_grad():
            condition, uncondition = self.conditioner(y)

        if self.eval_original_model:
            samples = self.diffusion_sampler(self.denoiser, xT, condition, uncondition)
        else:
            samples = self.diffusion_sampler(self.ema_denoiser, xT, condition, uncondition)

        samples = self.vae.decode(samples)
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx)
        return samples

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination,
            prefix=prefix+"denoiser.",
            keep_vars=keep_vars)
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=prefix+"ema_denoiser.",
            keep_vars=keep_vars)
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=prefix+"diffusion_trainer.",
            keep_vars=keep_vars)
        return destination


# ============================================================
#  CLI
# ============================================================

class ReWriteRootSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        stamp = time.strftime('%y%m%d%H%M')
        file_path = os.path.join(trainer.default_root_dir, f"config-{stage}-{stamp}.yaml")
        self.parser.save(
            self.config, file_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )


class BaseReWriteRootDirCli(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TagsClass:
            def __init__(self, exp:str):
                ...
        parser.add_class_arguments(TagsClass, nested_key="tags")

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--torch_hub_dir", type=str, default=None, help=("torch hub dir"),)
        parser.add_argument("--huggingface_cache_dir", type=str, default=None, help=("huggingface hub dir"),)
        parser.add_argument("--pretrained_ckpt_path", type=str, default=None, help="Path to pretrained checkpoint")
        parser.add_argument("--pretrained_ema", type=bool, default=False, help="Load ema weights")

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        config_trainer = self._get(self.config_init, "trainer", default={})
        default_root_dir = config_trainer.get("default_root_dir", None)

        if default_root_dir is None:
            default_root_dir = os.path.join(os.getcwd(), "workdirs")

        dirname = ""
        for v, k in self._get(self.config, "tags", default={}).items():
            dirname += f"{v}_{k}"
        default_root_dir = os.path.join(default_root_dir, dirname)

        self.resume_path = self._get(self.config_init, "ckpt_path", default=None)
        if not self.resume_path and os.path.exists(default_root_dir):
            ckpts = [ckpt for ckpt in os.listdir(default_root_dir) if ckpt.endswith('.ckpt')]
            if len(ckpts) > 0:
                def extract_step(f):
                    m = re.search(r"step=(\d+)", f)
                    return int(m.group(1)) if m else -1
                latest_ckpt = max(ckpts, key=extract_step)
                self.resume_path = os.path.join(default_root_dir, latest_ckpt)

                if os.path.exists(os.path.join(self.resume_path, "ddp_split.txt")):
                    merged_ckpt_path = os.path.join(self.resume_path, "checkpoint_merged.ckpt")
                    if not os.path.exists(merged_ckpt_path):
                        checkpoint = {}
                        for fname in os.listdir(self.resume_path):
                            if fname.startswith("checkpoint-") and fname.endswith(".pt"):
                                data = torch.load(os.path.join(self.resume_path, fname), map_location="cpu")
                                checkpoint.update(data)
                        torch.save(checkpoint, merged_ckpt_path)
                    self.resume_path = merged_ckpt_path

                print(f"[Resume Training] checkpoint from : {latest_ckpt}")

        config_trainer.default_root_dir = default_root_dir
        trainer = super().instantiate_trainer(**kwargs)
        if trainer.is_global_zero:
            os.makedirs(default_root_dir, exist_ok=True)
        if self.resume_path:
            trainer.ckpt_path = self.resume_path
        return trainer

    def _load_pretrained(self):
        raise NotImplementedError

    def instantiate_classes(self) -> None:
        torch_hub_dir = self._get(self.config, "torch_hub_dir")
        huggingface_cache_dir = self._get(self.config, "huggingface_cache_dir")
        if huggingface_cache_dir is not None:
            os.environ["HUGGINGFACE_HUB_CACHE"] = huggingface_cache_dir
        if torch_hub_dir is not None:
            os.environ["TORCH_HOME"] = torch_hub_dir
            torch.hub.set_dir(torch_hub_dir)
        super().instantiate_classes()

        if getattr(self, "resume_path", None) and (os.path.isfile(self.resume_path) or os.path.isfile(os.path.join(self.resume_path, "latest"))):
            print(f"Resume detected, skip pretrained_ckpt_path")
            return

        self._load_pretrained()


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
