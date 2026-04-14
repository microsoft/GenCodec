"""
Merge LoRA weights into base model.

Usage:
    python -m finetuned_one_step_codec.utils.merge_lora_stage1 \
        --input <lora_checkpoint.pt> \
        --output <merged_checkpoint.pt> \
        --config <config.yaml>
"""

import argparse
import torch
import torch.nn as nn
import yaml
from typing import Optional
from peft import LoraConfig, get_peft_model
from omegaconf import OmegaConf

from cod.utils.test_utils import instantiate_class, load_model
from finetuned_one_step_codec.main_stage1 import find_lora_target_modules


def merge_lora(
    input_ckpt_path: str,
    output_ckpt_path: str,
    config_path: str,
    lora_rank: int = 32,
    lora_alpha: Optional[int] = None,
    exclude_modules: list[str] = ["y_embedder", "conv2"],
    modules_to_save: list[str] = ["y_embedder", "conv2"],
) -> str:
    """Merge LoRA weights into the base model and save a clean checkpoint."""
    if lora_alpha is None:
        lora_alpha = lora_rank

    print(f"Merging LoRA weights")
    print(f"  Input: {input_ckpt_path}")
    print(f"  Output: {output_ckpt_path}")
    print(f"  Config: {config_path}")
    print(f"  LoRA Rank: {lora_rank}, Alpha: {lora_alpha}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.create(config)

    net = instantiate_class(config.model.net).cuda()

    lora_target_modules = find_lora_target_modules(
        net,
        target_types=(nn.Linear, nn.Conv2d, nn.Conv1d),
        exclude_modules=exclude_modules
    )
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=0.0, bias="none",
        modules_to_save=modules_to_save
    )
    net = get_peft_model(net, lora_config)

    ckpt = torch.load(input_ckpt_path, map_location="cpu")
    net = load_model(ckpt, net, prefix="net.")

    net.merge_and_unload()
    merged = net.base_model.model

    prefix = "net."
    sd = merged.state_dict()
    new_sd = {}
    for k in sd:
        if "modules_to_save.default." in k:
            special_key = k.replace("modules_to_save.default.", "")
            new_sd[prefix + special_key] = sd[k]
            continue
        if "original_module." in k:
            continue
        new_sd[prefix + k] = sd[k]

    # Preserve non-net weights from original checkpoint
    for k in ckpt['state_dict']:
        if not k.startswith(prefix):
            new_sd[k] = ckpt['state_dict'][k]

    torch.save(new_sd, output_ckpt_path)
    print(f"Merged checkpoint saved to {output_ckpt_path}")
    return output_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input LoRA checkpoint path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output merged checkpoint path")
    parser.add_argument("--config", "-c", type=str, required=True, help="Model config path")
    parser.add_argument("--lora-rank", "-r", type=int, default=32, help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", "-a", type=int, default=None, help="LoRA alpha (default: same as rank)")
    args = parser.parse_args()

    merge_lora(
        input_ckpt_path=args.input,
        output_ckpt_path=args.output,
        config_path=args.config,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
