import subprocess
import os
import re
import glob
import argparse
import yaml
import torch


STAGES = [
    {"name": "stage1_res_512", "stage": 1, "res": 512, "total_bs": 16},
    {"name": "stage2_res_512", "stage": 2, "res": 512, "total_bs": 32},
]

# Per-bpp codec parameters: (ds, codebook_bits)
BPP_PARAMS = {
    "0_0039": {"ds": 32, "codebook_bits": 4},
    "0_0078": {"ds": 32, "codebook_bits": 8},
    "0_0156": {"ds": 16, "codebook_bits": 4},
    "0_0312": {"ds": 16, "codebook_bits": 8},
    "0_1250": {"ds": 8,  "codebook_bits": 8},
    "0_5000": {"ds": 4,  "codebook_bits": 8},
}


def find_latest_checkpoint(exp_dir):
    ckpts = glob.glob(os.path.join(exp_dir, "**", "*.ckpt"), recursive=True)
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(m.group(1)) if (m := re.search(r"step[=\-](\d+)", p)) else 0)


def find_latest_checkpoint_dir(exp_dir):
    """Find latest .ckpt directory (stage1 saves checkpoints as directories)."""
    ckpt_dirs = [d for d in glob.glob(os.path.join(exp_dir, "**", "*.ckpt"), recursive=True)
                 if os.path.isdir(d)]
    if not ckpt_dirs:
        return find_latest_checkpoint(exp_dir)
    return max(ckpt_dirs, key=lambda p: int(m.group(1)) if (m := re.search(r"step[=\-](\d+)", p)) else 0)


def run_train(stage_dir, config_path, overrides, entry_script):
    """Run a single training subprocess."""
    cmd = ["PYTHONPATH=./", "python", entry_script, "fit", "-c", config_path]
    cmd += [f"--{k}={v}" for k, v in overrides.items()]

    log_i = 0
    while os.path.exists(os.path.join(stage_dir, f"train_{log_i}.log")):
        log_i += 1
    log_path = os.path.join(stage_dir, f"train_{log_i}.log")

    cmd_str = ' '.join(f'"{c}"' if ' ' in c else c for c in cmd)
    subprocess.run(f"set -o pipefail; {cmd_str} 2>&1 | tee '{log_path}'",
                   shell=True, check=True, executable='/bin/bash')


def get_bpp_overrides(bpp):
    """Return per-bpp overrides for ds and codebook_bits."""
    params = BPP_PARAMS[bpp]
    return {
        "model.net.init_args.ds": params["ds"],
        "model.net.init_args.codebook_bits": params["codebook_bits"],
    }


def run_stage1(args, ngpu, exp_dir):
    """Stage 1: LoRA -> Merge -> Full FT, or direct Full FT."""
    stage = STAGES[0]
    name = stage["name"]
    res = stage["res"]
    bs = max(stage["total_bs"] // ngpu, 1)
    acc = max(stage["total_bs"] // (bs * ngpu), 1)

    # Load stage 1 config template
    template = "finetuned_one_step_codec/configs/stage1.yaml"
    with open(template, "r") as f:
        raw_cfg = yaml.safe_load(f)
    lora_steps = raw_cfg.get("lora_steps")
    full_ft_steps = raw_cfg.get("full_ft_steps")

    cfg = raw_cfg
    meta = cfg["data"]["train_dataset"]["init_args"]["metadata"]
    test_root = cfg["data"]["eval_dataset"]["init_args"]["root"]

    def build_overrides(tags_exp, net_ckpt, pretrain_prefix, max_steps=None):
        ov = {
            "trainer.default_root_dir": exp_dir,
            "trainer.accumulate_grad_batches": acc,
            "data.train_dataset.init_args.root": args.data_dir,
            "data.train_dataset.init_args.metadata": os.path.join(args.data_dir, meta),
            "data.eval_dataset.init_args.root": os.path.join(args.data_dir, test_root),
            "data.pred_dataset.init_args.root": os.path.join(args.data_dir, test_root),
            "data.train_batch_size": bs,
            "tags.exp": tags_exp,
            "model.net.init_args.net_ckpt_path": net_ckpt,
            "model.net.init_args.pretrain_prefix": pretrain_prefix,
            "model.codec_trainer.init_args.net_loss.init_args.net_loss_ckpt_path": args.cod_ckpt,
        }
        ov.update(get_bpp_overrides(args.bpp))
        if args.dinov2_path:
            ov["model.codec_trainer.init_args.net_loss.init_args.encoder.init_args.weight_path"] = args.dinov2_path
        if max_steps is not None:
            ov["trainer.max_steps"] = max_steps
        return ov

    def save_config(stage_dir, use_lora):
        cfg_copy = dict(cfg)
        cfg_copy.pop("lora_steps", None)
        cfg_copy.pop("full_ft_steps", None)
        if not use_lora:
            cfg_copy.setdefault("model", {})["use_lora"] = False
        # Apply bpp overrides directly into config (CLI overrides may not work reliably)
        bpp_params = BPP_PARAMS[args.bpp]
        cfg_copy["model"]["net"]["init_args"]["ds"] = bpp_params["ds"]
        cfg_copy["model"]["net"]["init_args"]["codebook_bits"] = bpp_params["codebook_bits"]
        path = os.path.join(stage_dir, "config.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg_copy, f)
        return path

    if args.use_lora:
        # Phase 1: LoRA training
        lora_dir = os.path.join(exp_dir, f"exp_{name}_lora")
        os.makedirs(lora_dir, exist_ok=True)

        if os.path.exists(os.path.join(lora_dir, ".done")):
            print(f"[skip] {name}_lora already complete")
        else:
            print(f"\n[stage1/lora] res={res} bs={bs}x{ngpu}x{acc}={bs*ngpu*acc} steps={lora_steps}")
            overrides = build_overrides(f"{name}_lora", args.cod_ckpt, "ema_denoiser.", lora_steps)
            config_path = save_config(lora_dir, use_lora=True)
            run_train(lora_dir, config_path, overrides, "finetuned_one_step_codec/main_stage1.py")
            with open(os.path.join(lora_dir, ".done"), "w") as f:
                f.write("done\n")

        # Phase 2: Merge LoRA weights
        ckpt_dir = find_latest_checkpoint_dir(lora_dir)
        if ckpt_dir is None:
            raise RuntimeError(f"LoRA checkpoint not found in {lora_dir}")

        merged_ckpt = os.path.join(ckpt_dir, "checkpoint-state_dict-merged.pt")
        if not os.path.exists(merged_ckpt):
            print(f"\n[stage1/merge] Merging LoRA weights...")
            input_ckpt = os.path.join(ckpt_dir, "checkpoint-state_dict.pt")
            cmd = (f'PYTHONPATH=./ python -m finetuned_one_step_codec.utils.merge_lora_stage1 '
                   f'--input "{input_ckpt}" --output "{merged_ckpt}" '
                   f'--config "{template}" --lora-rank {args.lora_rank}')
            subprocess.run(cmd, shell=True, check=True)
        else:
            print(f"[skip] LoRA merge already done")

        # Phase 3: Full FT from merged weights
        ft_dir = os.path.join(exp_dir, f"exp_{name}")
        os.makedirs(ft_dir, exist_ok=True)

        if os.path.exists(os.path.join(ft_dir, ".done")):
            print(f"[skip] {name} already complete")
        else:
            print(f"\n[stage1/ft] res={res} bs={bs}x{ngpu}x{acc}={bs*ngpu*acc} steps={full_ft_steps}")
            overrides = build_overrides(name, merged_ckpt, "net.", full_ft_steps)
            config_path = save_config(ft_dir, use_lora=False)
            run_train(ft_dir, config_path, overrides, "finetuned_one_step_codec/main_stage1.py")
            with open(os.path.join(ft_dir, ".done"), "w") as f:
                f.write("done\n")
    else:
        # Direct full FT (no LoRA)
        ft_dir = os.path.join(exp_dir, f"exp_{name}")
        os.makedirs(ft_dir, exist_ok=True)

        if os.path.exists(os.path.join(ft_dir, ".done")):
            print(f"[skip] {name} already complete")
        else:
            print(f"\n[stage1/ft] res={res} bs={bs}x{ngpu}x{acc}={bs*ngpu*acc} steps={full_ft_steps}")
            overrides = build_overrides(name, args.cod_ckpt, "ema_denoiser.", full_ft_steps)
            config_path = save_config(ft_dir, use_lora=False)
            run_train(ft_dir, config_path, overrides, "finetuned_one_step_codec/main_stage1.py")
            with open(os.path.join(ft_dir, ".done"), "w") as f:
                f.write("done\n")

    return os.path.join(exp_dir, f"exp_{name}")


def run_stage2(args, ngpu, exp_dir, stage1_ckpt):
    """Stage 2: DMD + discriminator training."""
    stage = STAGES[1]
    name = stage["name"]
    res = stage["res"]
    bs = max(stage["total_bs"] // ngpu, 1)
    acc = max(stage["total_bs"] // (bs * ngpu), 1)

    stage_dir = os.path.join(exp_dir, f"exp_{name}")
    os.makedirs(stage_dir, exist_ok=True)

    if os.path.exists(os.path.join(stage_dir, ".done")):
        print(f"[skip] {name} already complete")
        return stage_dir

    # Load and resolve config
    template = "finetuned_one_step_codec/configs/stage2.yaml"
    with open(template, "r") as f:
        cfg = yaml.safe_load(f)

    meta = cfg["data"]["train_dataset"]["init_args"]["metadata"]
    test_root = cfg["data"]["eval_dataset"]["init_args"]["root"]

    overrides = {
        "trainer.default_root_dir": exp_dir,
        "model.grad_accum_steps": acc,
        "data.train_dataset.init_args.root": args.data_dir,
        "data.train_dataset.init_args.metadata": os.path.join(args.data_dir, meta),
        "data.eval_dataset.init_args.root": os.path.join(args.data_dir, test_root),
        "data.pred_dataset.init_args.root": os.path.join(args.data_dir, test_root),
        "data.train_batch_size": bs,
        "tags.exp": name,
        "model.net.init_args.net_ckpt_path": stage1_ckpt,
        "model.dmd_trainer.init_args.net_loss.init_args.net_loss_ckpt_path": stage1_ckpt,
        "dmd_ckpt_path": args.dmd_ckpt,
    }
    overrides.update(get_bpp_overrides(args.bpp))
    if args.dinov2_path:
        overrides["model.dmd_trainer.init_args.net_loss.init_args.encoder.init_args.weight_path"] = args.dinov2_path
        overrides["model.disc_weight_path"] = args.dinov2_path

    # Apply bpp overrides directly into config
    bpp_params = BPP_PARAMS[args.bpp]
    cfg["model"]["net"]["init_args"]["ds"] = bpp_params["ds"]
    cfg["model"]["net"]["init_args"]["codebook_bits"] = bpp_params["codebook_bits"]

    # Save resolved config
    config_path = os.path.join(stage_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"\n[stage2] {name}  res={res} bs={bs}x{ngpu}x{acc}={bs*ngpu*acc}")
    run_train(stage_dir, config_path, overrides, "finetuned_one_step_codec/main_stage2.py")

    with open(os.path.join(stage_dir, ".done"), "w") as f:
        f.write("done\n")
    return stage_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuned One-Step CoD-Lite training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset root containing anno_512/, test/")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Parent directory for experiment outputs")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name (creates exp_<name>/ under save_dir)")
    parser.add_argument("--bpp", type=str, default="0_0039",
                        choices=list(BPP_PARAMS.keys()),
                        help="Bits per pixel, underscore for decimal (default: 0_0039)")
    parser.add_argument("--cod_ckpt", type=str, required=True,
                        help="Pretrained CoD checkpoint path")
    parser.add_argument("--dmd_ckpt", type=str, required=True,
                        help="Pretrained DMD checkpoint path")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA -> merge -> full FT pipeline (default: direct full FT)")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--dinov2_path", type=str, default=None,
                        help="Path to DINOv2 weights (optional)")
    args = parser.parse_args()

    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "No GPU detected"
    exp_dir = os.path.join(args.save_dir, f"exp_{args.exp_name}")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"One-Step CoD-Lite: {args.exp_name}  bpp={args.bpp}  lora={args.use_lora}  gpus={ngpu}")

    try:
        stage1_dir = run_stage1(args, ngpu, exp_dir)

        # Find stage1 final checkpoint for stage2
        stage1_ckpt = find_latest_checkpoint_dir(stage1_dir)
        if stage1_ckpt is None:
            raise RuntimeError(f"No checkpoint found in {stage1_dir}")
        print(f"\nStage1 checkpoint: {stage1_ckpt}")

        run_stage2(args, ngpu, exp_dir, stage1_ckpt)
    except subprocess.CalledProcessError as e:
        print(f"\nFailed (exit {e.returncode})")

    print("\nDone.")
