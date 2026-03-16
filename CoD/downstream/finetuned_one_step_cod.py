"""
One-Step CoD Inference: compress, decompress, or evaluate.

Download:
    huggingface-cli download jzyustc/CoD --include "finetuned_one_step_cod/*" --local-dir ./pretrained/CoD

Usage:
    python -m downstream.finetuned_one_step_cod evaluate \
        --ckpt ./pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.pt \
        --config ./pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.yaml \
        --input <image_dir> --output <recon_dir>
"""

import os
import argparse
import struct
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from PIL import Image

from cod.models.pixel_dit import PixNerDiT, FlattenDiTBlock, modulate


# ============================================================
#  Model
# ============================================================

class CacheFlattenDiTBlock(nn.Module):
    """Pre-compute and cache adaLN modulation for FlattenDiTBlock at t=0."""

    def __init__(self, block: FlattenDiTBlock):
        super().__init__()
        self.block = block
        self.shift_msa = None
        self.scale_msa = None
        self.gate_msa = None
        self.shift_mlp = None
        self.scale_mlp = None
        self.gate_mlp = None

    @torch.no_grad()
    def convert_cache(self, c):
        (self.shift_msa, self.scale_msa, self.gate_msa,
         self.shift_mlp, self.scale_mlp, self.gate_mlp) = self.block.adaLN_modulation(c).chunk(6, dim=-1)
        del self.block.adaLN_modulation

    def forward(self, x, c, pos, mask=None):
        x = x + self.gate_msa * self.block.attn(
            modulate(self.block.norm1(x), self.shift_msa, self.scale_msa), pos, mask=mask)
        x = x + self.gate_mlp * self.block.mlp(
            modulate(self.block.norm2(x), self.shift_mlp, self.scale_mlp))
        return x


def _replace_with_cache(module, c):
    for name, child in module.named_children():
        if isinstance(child, FlattenDiTBlock):
            cached = CacheFlattenDiTBlock(child)
            cached.convert_cache(c)
            setattr(module, name, cached)
        else:
            _replace_with_cache(child, c)


class OneStepCoD(PixNerDiT):
    """One-step CoD: single forward pass at t=0 with optional input noise."""

    def __init__(self, noise_level=0.0, pred='x', **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level
        self.pred = pred
        self._cached = False

    def eval(self):
        if not self._cached:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            t = torch.zeros((1,), device=device, dtype=dtype)
            c = nn.functional.silu(self.t_embedder(t).view(1, -1, self.hidden_size))
            _replace_with_cache(self, c)
            self._cached = True
        return super().eval()

    @torch.no_grad()
    def inference(self, y, cond=None):
        """
        One-step image reconstruction.

        Args:
            y: Input image tensor [B, 3, H, W] in [0, 1] range.
            cond: Pre-computed codec condition (optional).

        Returns:
            Reconstructed image [B, 3, H, W] in [-1, 1] range.
        """
        b, c, h, w = y.shape
        t = torch.zeros((b,), device=y.device, dtype=y.dtype)
        noise = torch.randn_like(y) * self.noise_level
        out, cond = self.forward(noise, t, y, cond=cond)

        if self.pred == 'v':
            out = noise + out
        # pred='x': out is direct prediction
        return out


# ============================================================
#  Load Model
# ============================================================

def load_one_step_model(ckpt_path, config_path, device="cuda"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    ckpt_prefix = cfg["ckpt_prefix"]
    init_args = cfg["model"]["init_args"]

    net = OneStepCoD(**init_args).to(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip prefix and load
    net_dict = {}
    for k, v in state_dict.items():
        if k.startswith(ckpt_prefix):
            net_dict[k[len(ckpt_prefix):]] = v

    missing, unexpected = net.load_state_dict(net_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"  {k}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  {k}")

    net.eval()
    print(f"Loaded {config_path} from {ckpt_path} (prefix={ckpt_prefix})")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")
    return net


def fp2uint8(x):
    """Convert [-1, 1] float tensor to [0, 255] uint8."""
    return torch.clip_((x + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)


# ============================================================
#  Compress
# ============================================================

def write_cod_file(bitstream, width, height, output_path):
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<H', width))
        f.write(struct.pack('<H', height))
        f.write(bytes(bitstream))


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def do_compress(net, input_path, output_path):
    image_files = sorted([
        os.path.join(input_path, f) for f in os.listdir(input_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    pbar = tqdm(total=len(image_files), desc="Compressing")

    for image_file in image_files:
        y = to_tensor(Image.open(image_file).convert("RGB")).to("cuda").unsqueeze(0)
        H, W = y.shape[-2:]
        bitstream = net.compress(y)

        out_path = os.path.join(output_path, f"{os.path.basename(image_file)}.cod")
        write_cod_file(bitstream, H, W, out_path)
        pbar.update(1)


# ============================================================
#  Decompress
# ============================================================

def read_cod_file(file_path):
    with open(file_path, 'rb') as f:
        width = struct.unpack('<H', f.read(2))[0]
        height = struct.unpack('<H', f.read(2))[0]
        bitstream = list(f.read())
    return width, height, bitstream


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def do_decompress(net, input_path, output_path):
    bin_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path)])
    pbar = tqdm(total=len(bin_files), desc="Decompressing")

    for bin_file in bin_files:
        W, H, bitstream = read_cod_file(bin_file)
        codec_cond = net.decompress(bitstream, H, W, "cuda")
        img = net.inference(
            y=torch.zeros((1, 3, H, W), device=codec_cond.device, dtype=codec_cond.dtype),
            cond=codec_cond
        )
        img = fp2uint8(img)

        out_img = Image.fromarray(img[0].permute(1, 2, 0).cpu().numpy())
        out_path = os.path.join(output_path, os.path.basename(bin_file).replace('.cod', ''))
        out_img.save(out_path)
        pbar.update(1)


# ============================================================
#  Evaluate (compress + decompress in one pass)
# ============================================================

@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def do_evaluate(net, input_path, output_path):
    image_files = sorted([
        os.path.join(input_path, f) for f in os.listdir(input_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    pbar = tqdm(total=len(image_files), desc="Evaluating")

    for image_file in image_files:
        y = to_tensor(Image.open(image_file).convert("RGB")).to("cuda").unsqueeze(0)
        img = net.inference(y)
        img = fp2uint8(img)

        out_img = Image.fromarray(img[0].permute(1, 2, 0).cpu().numpy())
        out_path = os.path.join(output_path, os.path.basename(image_file))
        out_img.save(out_path)
        pbar.update(1)


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-Step CoD inference")
    parser.add_argument("mode", choices=["compress", "decompress", "evaluate"])
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    net = load_one_step_model(args.ckpt, config_path=args.config, device=args.device)

    if args.mode == "compress":
        do_compress(net, args.input, args.output)
    elif args.mode == "decompress":
        do_decompress(net, args.input, args.output)
    elif args.mode == "evaluate":
        do_evaluate(net, args.input, args.output)
