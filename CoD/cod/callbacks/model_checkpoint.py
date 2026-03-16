import os.path
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy


class CheckpointHook(ModelCheckpoint):
    """Save checkpoint with only the incremental part of the model"""
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self.dirpath = trainer.default_root_dir
        self.exception_ckpt_path = os.path.join(self.dirpath, "on_exception.pt")
        pl_module.strict_loading = False

    def on_save_checkpoint(
            self, trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            checkpoint: Dict[str, Any]
    ) -> None:
        del checkpoint["callbacks"]

    # def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
    #     if not "debug" in self.exception_ckpt_path:
    #         trainer.save_checkpoint(self.exception_ckpt_path)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        if isinstance(trainer.strategy, DDPStrategy):
            self._last_global_step_saved = trainer.global_step
            self._last_checkpoint_saved = filepath

            step = trainer.global_step
            epoch = trainer.current_epoch
            ckpt_name = f"epoch={epoch}-step={step}.ckpt"
            ckpt_path = os.path.join(self.dirpath, ckpt_name)
            os.makedirs(ckpt_path, exist_ok=True)
            checkpoint = trainer._checkpoint_connector.dump_checkpoint(False)
            for k, v in checkpoint.items():
                torch.save({k: v}, os.path.join(ckpt_path, f"checkpoint-{k}.pt"))
            with open(os.path.join(ckpt_path, "ddp_split.txt"), "w") as f:
                pass
        else:
            super()._save_checkpoint(trainer, filepath)
