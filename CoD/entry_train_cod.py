import subprocess
import os
import re
import glob
import argparse
import yaml
import torch

# target_total_bs = bs_per_gpu * ngpu * accumulation
STAGES = [
    {"name": "stage1_res_256",              "res": 256, "total_bs": 128, "unified_post": False, "ckpt": None},
    {"name": "stage2_res_512",              "res": 512, "total_bs": 64,  "unified_post": False, "ckpt": "auto"},
    {"name": "stage3_res_512_unified_post", "res": 512, "total_bs": 64,  "unified_post": True,  "ckpt": "auto"},
]


def find_latest_checkpoint(exp_dir):
    ckpts = glob.glob(os.path.join(exp_dir, "**", "*.ckpt"), recursive=True)
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(m.group(1)) if (m := re.search(r"step[=\-](\d+)", p)) else 0)


def run_stage(stage, exp_dir, bpp, latent, ngpu, data_dir, dinov2_path, prev_dir=None):
    name = stage["name"]
    res = stage["res"]
    stage_dir = os.path.join(exp_dir, f"exp_{name}")
    os.makedirs(stage_dir, exist_ok=True)

    # Check if already complete
    if os.path.exists(os.path.join(stage_dir, ".done")):
        print(f"[skip] {name} already complete")
        return stage_dir

    # Resolve checkpoint from previous stage
    ckpt = stage["ckpt"]
    if ckpt == "auto" and prev_dir:
        ckpt = find_latest_checkpoint(prev_dir)

    # Batch size
    bs = max(stage["total_bs"] // ngpu, 1)
    acc = max(stage["total_bs"] // (bs * ngpu), 1)

    # Load template config
    mode = "latent" if latent else "pixel"
    suffix = "_unified_post" if stage["unified_post"] else ""
    template = f"cod/configs/{mode}/bpp_{bpp}_pix{res}_xl{suffix}.yaml"
    with open(template, "r") as f:
        cfg = yaml.safe_load(f)

    meta = cfg["data"]["train_dataset"]["init_args"]["metadata"]
    test_root = cfg["data"]["eval_dataset"]["init_args"]["root"]

    # Build overrides
    overrides = {
        "trainer.default_root_dir": exp_dir,
        "trainer.accumulate_grad_batches": acc,
        "data.train_dataset.init_args.root": data_dir,
        "data.train_dataset.init_args.metadata": os.path.join(data_dir, meta),
        "data.eval_dataset.init_args.root": os.path.join(data_dir, test_root),
        "data.pred_dataset.init_args.root": os.path.join(data_dir, test_root),
        "data.train_batch_size": bs,
        "tags.exp": name,
    }
    if dinov2_path:
        overrides["model.diffusion_trainer.init_args.encoder.init_args.weight_path"] = dinov2_path
    if ckpt and ckpt != "auto":
        overrides["pretrained_ckpt_path"] = ckpt

    # Save template config for reference
    config_path = os.path.join(stage_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    # Run training
    cmd = ["PYTHONPATH=./", "python", "cod/main.py", "fit", "-c", config_path]
    cmd += [f"--{k}={v}" for k, v in overrides.items()]

    print(f"\n[stage] {name}  res={res} bs={bs}x{ngpu}x{acc}={bs*ngpu*acc}")
    log_i = 0
    while os.path.exists(os.path.join(stage_dir, f"train_{log_i}.log")):
        log_i += 1
    log_path = os.path.join(stage_dir, f"train_{log_i}.log")
    cmd_str = ' '.join(f'"{c}"' if ' ' in c else c for c in cmd)
    subprocess.run(f"set -o pipefail; {cmd_str} 2>&1 | tee '{log_path}'",
                   shell=True, check=True, executable='/bin/bash')

    # Mark complete
    with open(os.path.join(stage_dir, ".done"), "w") as f:
        f.write("done\n")
    return stage_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoD multi-stage training")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--bpp", type=str, default="0_0039")
    parser.add_argument("--latent", action="store_true")
    parser.add_argument("--dinov2_path", type=str, default=None)
    args = parser.parse_args()

    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "No GPU detected"
    exp_dir = os.path.join(args.save_dir, f"exp_{args.exp_name}")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"CoD Training: {args.exp_name}  bpp={args.bpp}  latent={args.latent}  gpus={ngpu}")

    prev_dir = None
    for stage in STAGES:
        try:
            prev_dir = run_stage(
                stage, exp_dir, args.bpp, args.latent,
                ngpu, args.data_dir, args.dinov2_path, prev_dir,
            )
        except subprocess.CalledProcessError as e:
            print(f"\nFailed at {stage['name']} (exit {e.returncode})")
            break

    print("\nDone.")
