import os
import shutil
import threading
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy


def _move_to_cpu(obj):
    """Recursively move all tensors to CPU and detach."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_cpu(v) for v in obj)
    return obj


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _write_to_dir(ckpt_dir, checkpoint):
    """Write checkpoint parts to a directory."""
    for k, v in checkpoint.items():
        torch.save({k: v}, os.path.join(ckpt_dir, f"checkpoint-{k}.pt"))
    with open(os.path.join(ckpt_dir, "ddp_split.txt"), "w") as f:
        pass


def _bg_write_local(ckpt_dir, checkpoint):
    """Background: torch.save to local filesystem."""
    try:
        _write_to_dir(ckpt_dir, checkpoint)
        if _get_rank() == 0:
            print(f"  [async ckpt] saved: {ckpt_dir}")
    except Exception as e:
        print(f"  [async ckpt] FAILED write {ckpt_dir}: {e}")


class CheckpointHook(ModelCheckpoint):
    """Save checkpoint with only the incremental part of the model, async IO."""

    def __init__(self, *args, keep_every_n_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._save_thread: Optional[threading.Thread] = None
        self._keep_every_n_steps = keep_every_n_steps
        self._saved_ckpt_dirs: dict = {}

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self.dirpath = trainer.default_root_dir
        pl_module.strict_loading = False

    def on_save_checkpoint(
            self, trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint.pop("callbacks", None)

    def _wait_prev_save(self):
        if self._save_thread is not None and self._save_thread.is_alive():
            if _get_rank() == 0:
                print(f"  [async ckpt] waiting for previous save...")
            self._save_thread.join()

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        if isinstance(trainer.strategy, DDPStrategy):
            self._wait_prev_save()

            self._last_global_step_saved = trainer.global_step
            self._last_checkpoint_saved = filepath

            step = trainer.global_step
            epoch = trainer.current_epoch
            ckpt_name = f"epoch={epoch}-step={step}.ckpt"
            ckpt_dir = os.path.join(self.dirpath, ckpt_name)

            checkpoint = trainer._checkpoint_connector.dump_checkpoint(False)
            checkpoint = _move_to_cpu(checkpoint)
            torch.cuda.synchronize()

            os.makedirs(ckpt_dir, exist_ok=True)
            self._save_thread = threading.Thread(
                target=_bg_write_local,
                args=(ckpt_dir, checkpoint),
                daemon=True,
            )
            self._save_thread.start()
            if _get_rank() == 0:
                print(f"  [async ckpt] started: {ckpt_name}")

            self._saved_ckpt_dirs[step] = ckpt_dir
            self._cleanup_old_checkpoints(step)
        else:
            super()._save_checkpoint(trainer, filepath)

    def _cleanup_old_checkpoints(self, current_step):
        """Delete non-milestone checkpoints, keeping only keep_every_n_steps multiples + current."""
        if self._keep_every_n_steps is None:
            return

        to_delete = []
        for step, ckpt_dir in self._saved_ckpt_dirs.items():
            if step == current_step:
                continue
            if step % self._keep_every_n_steps == 0:
                continue
            to_delete.append(step)

        for step in to_delete:
            ckpt_dir = self._saved_ckpt_dirs.pop(step)
            if _get_rank() == 0:
                threading.Thread(
                    target=lambda d: shutil.rmtree(d, ignore_errors=True),
                    args=(ckpt_dir,),
                    daemon=True,
                ).start()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._wait_prev_save()
        super().on_train_end(trainer, pl_module)

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._wait_prev_save()
        super().teardown(trainer, pl_module, stage)
