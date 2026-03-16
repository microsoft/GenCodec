"""
CoD as Perceptual Loss: MS-ILLM decoder finetuned with CoD-based DMD loss.

Download:
    huggingface-cli download jzyustc/CoD --include "perceptual_loss_illm_dec/*" --local-dir ./pretrained/CoD

Usage:
    python -m downstream.perceptual_loss_inference \
        --ckpt ./pretrained/CoD/perceptual_loss_illm_dec/msillm_quality_1.pt \
        --quality 1 \
        --input <image_dir> \
        --output <recon_dir>

Available quality levels: 1, 2, 3, 4, vlo2
"""

import os
import argparse
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np


def load_msillm_finetuned(ckpt_path, quality="1", device="cuda"):
    """Load MS-ILLM with finetuned decoder from CoD DMD training."""
    model_name = f"msillm_quality_{quality}"

    # Load base MS-ILLM model from NeuralCompression
    codec = torch.hub.load("facebookresearch/NeuralCompression", model_name)

    # Load finetuned decoder weights
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Try both checkpoint prefixes: "net.codec." (quality 1-4) and "codec." (vlo2)
    for prefix in ("net.codec.", "codec."):
        decoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                decoder_dict[k[len(prefix):]] = v
        if decoder_dict:
            break

    missing, unexpected = codec.load_state_dict(decoder_dict, strict=False)
    loaded = len(decoder_dict) - len(unexpected)
    print(f"Loaded {loaded} finetuned parameters into {model_name}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    codec = codec.to(device).eval()
    return codec


@torch.no_grad()
def evaluate(codec, input_path, output_path, device="cuda"):
    image_files = sorted([
        os.path.join(input_path, f) for f in os.listdir(input_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    pbar = tqdm(total=len(image_files), desc="Evaluating")

    for image_file in image_files:
        img = to_tensor(Image.open(image_file).convert("RGB")).unsqueeze(0).to(device)
        out = codec(img)
        recon = out.image.clamp(0, 1)
        recon = (recon[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        out_img = Image.fromarray(recon)
        out_path = os.path.join(output_path, os.path.basename(image_file))
        out_img.save(out_path)
        pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MS-ILLM with CoD-finetuned decoder")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to finetuned checkpoint")
    parser.add_argument("--quality", type=str, default="1", choices=["1", "2", "3", "4", "vlo2"],
                        help="MS-ILLM quality level")
    parser.add_argument("--input", type=str, required=True, help="Input image directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    codec = load_msillm_finetuned(args.ckpt, quality=args.quality, device=args.device)
    evaluate(codec, args.input, args.output, device=args.device)
