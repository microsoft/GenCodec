"""
One-Step CoD Inference: compress, decompress, or evaluate.

Usage:
    python -m finetuned_one_step_codec.inference compress \
        --ckpt <checkpoint> --config <config.yaml> --input <image_dir> --output <output_dir>

    python -m finetuned_one_step_codec.inference decompress \
        --ckpt <checkpoint> --config <config.yaml> --input <cod_dir> --output <output_dir>

    python -m finetuned_one_step_codec.inference evaluate \
        --ckpt <checkpoint> --config <config.yaml> --input <image_dir> --output <output_dir>
"""

import os
import argparse
import struct
import torch
import yaml
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from PIL import Image
from omegaconf import OmegaConf
from cod.models.autoencoder import fp2uint8
from cod.utils.test_utils import instantiate_class, load_model


def write_cod_file(bitstream, width, height, output_path):
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<H', width))
        f.write(struct.pack('<H', height))
        f.write(bytes(bitstream))


def read_cod_file(file_path):
    with open(file_path, 'rb') as f:
        width = struct.unpack('<H', f.read(2))[0]
        height = struct.unpack('<H', f.read(2))[0]
        bitstream = list(f.read())
    return width, height, bitstream


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def compress(net, image_path, output_path):
    image_files = sorted([
        os.path.join(image_path, f) for f in os.listdir(image_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    pbar = tqdm(total=len(image_files), desc="Compressing")
    for image_file in image_files:
        y = to_tensor(Image.open(image_file).convert("RGB")).to('cuda').unsqueeze(0)
        H, W = y.shape[-2:]
        bitstream = net.compress(y)
        out_path = os.path.join(output_path, f"{os.path.basename(image_file)}.cod")
        write_cod_file(bitstream, W, H, out_path)
        pbar.update(1)


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def decompress(net, input_path, output_path):
    bin_files = sorted(os.path.join(input_path, f) for f in os.listdir(input_path))
    pbar = tqdm(total=len(bin_files), desc="Decompressing")
    for bin_file in bin_files:
        W, H, bitstream = read_cod_file(bin_file)
        codec_cond = net.decompress(bitstream, H, W, "cuda")
        img = net.inference(
            y=torch.zeros((1, 3, H, W), device=codec_cond.device, dtype=codec_cond.dtype),
            cond=codec_cond)
        img = fp2uint8(img)
        out_img = Image.fromarray(img[0].permute(1, 2, 0).cpu().numpy())
        out_path = os.path.join(output_path, os.path.basename(bin_file).replace('.cod', ''))
        out_img.save(out_path)
        pbar.update(1)


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def evaluate(net, image_path, output_path):
    image_files = sorted([
        os.path.join(image_path, f) for f in os.listdir(image_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    pbar = tqdm(total=len(image_files), desc="Evaluating")
    for image_file in image_files:
        y = to_tensor(Image.open(image_file).convert("RGB")).to('cuda').unsqueeze(0)
        img = net.inference(y)
        img = fp2uint8(img)
        out_img = Image.fromarray(img[0].permute(1, 2, 0).cpu().numpy())
        out_path = os.path.join(output_path, os.path.basename(image_file))
        out_img.save(out_path)
        pbar.update(1)


def load_net(ckpt_path, config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.create(config)

    net = instantiate_class(config.model.net).cuda()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = load_model(ckpt, net, prefix="net.")
    net.eval()
    print(f"[Parameters]: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-Step CoD inference")
    parser.add_argument("mode", choices=["compress", "decompress", "evaluate"])
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    net = load_net(args.ckpt, args.config)

    if args.mode == "compress":
        compress(net, args.input, args.output)
    elif args.mode == "decompress":
        decompress(net, args.input, args.output)
    elif args.mode == "evaluate":
        evaluate(net, args.input, args.output)
