import os
import argparse
import torch
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ref = f"/path/to/gt/"
    recons = [
        "/path/to/recon",
    ]

    n_gpus = 1 #torch.cuda.device_count()
    procs = []
    for i, recon in enumerate(recons):
        out_name = "metric.csv"

        gpu_id = i % n_gpus 
        cmd = [
            "python", "scripts/metric.py",
            "--ref", ref,
            "--recon", recon,
            "--device", f"cuda:{gpu_id}",
            "--fid_patch_size", "64",       # 64 for images at 512x512
            "--fid_patch_num", "2",
            "--output_path", f'{recon}/../',
            "--output_name", out_name
        ]
        print("Launching on GPU", gpu_id, ":", " ".join(cmd))
        procs.append(subprocess.Popen(cmd))

        if (i + 1) % n_gpus == 0:
            for p in procs:
                p.wait()
            procs = []

    for p in procs:
        p.wait()
