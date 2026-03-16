import os
import argparse
import struct
from tqdm import tqdm
import torch
from torchvision.transforms.functional import to_tensor
import yaml
from omegaconf import OmegaConf
from PIL import Image
from cod.models.autoencoder import fp2uint8
from cod.diffusion import simple_guidance_fn, AdamLMSampler, EulerSampler, LinearScheduler
from cod.utils.test_utils import instantiate_class, load_model


class Pipeline:
    def __init__(self, vae, denoiser, conditioner, latent):
        self.vae = vae
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.latent = latent


def load_pipeline(config_path, ckpt_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.create(config)

    vae = instantiate_class(config.model.vae).cuda()
    denoiser = instantiate_class(config.model.denoiser).cuda()
    conditioner = instantiate_class(config.model.conditioner).cuda()
    latent = config.model.latent

    ckpt = torch.load(ckpt_path, map_location="cpu")
    denoiser = load_model(ckpt, denoiser, prefix="ema_denoiser.")
    denoiser.eval()

    return Pipeline(vae, denoiser, conditioner, latent)


def create_sampler(sampler, step, cfg):
    # NOTE: adam2 for v-pred, euler for x-pred
    if sampler == "adam2":
        return AdamLMSampler(
            order=2,
            scheduler=LinearScheduler(),
            guidance_fn=simple_guidance_fn,
            num_steps=step,
            timeshift=1.0,
            guidance=cfg,
            guidance_interval_min=0.1,
            guidance_interval_max=1.0,
            last_step=0.04,
        )
    else:
        return EulerSampler(
            scheduler=LinearScheduler(),
            w_scheduler=LinearScheduler(),
            guidance_fn=simple_guidance_fn,
            num_steps=step,
            timeshift=1.0,
            guidance=cfg,
            guidance_interval_min=0.1,
            guidance_interval_max=1.0,
            last_step=0.04,
        )


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
def compress(pipeline, input_path, output_path):
    image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_files.sort()
    pbar = tqdm(total=len(image_files), desc="Compressing")

    for image_file in image_files:
        y = to_tensor(Image.open(image_file).convert("RGB")).to('cuda').unsqueeze(0)
        H, W = y.shape[-2:]
        cond, uncond = pipeline.conditioner(y)
        bitstream = pipeline.denoiser.compress(cond)

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
def decompress(pipeline, input_path, output_path, seed=0, step=25, cfg=1.0, sampler="adam2"):
    diffusion_sampler = create_sampler(sampler, step, cfg)

    bin_files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    bin_files.sort()
    pbar = tqdm(total=len(bin_files), desc="Decompressing")

    for bin_file in bin_files:
        W, H, bitstream = read_cod_file(bin_file)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        xT = torch.randn((1, 4, H // 8, W // 8) if pipeline.latent else (1, 3, H, W), device="cpu", dtype=torch.float32, generator=generator).to("cuda")

        _, uncond = pipeline.conditioner(torch.rand_like(xT))
        codec_cond = pipeline.denoiser.decompress(bitstream, H, W, "cuda", uncond=uncond)

        sample = diffusion_sampler(pipeline.denoiser, xT, None, None, codec_cond=codec_cond)
        img = pipeline.vae.decode(sample)
        img = fp2uint8(img)

        out_img = Image.fromarray(img[0].permute(1, 2, 0).cpu().numpy())
        out_path = os.path.join(output_path, os.path.basename(bin_file).replace('.cod', ''))
        out_img.save(out_path)
        pbar.update(1)


# ============================================================
#  Evaluate (encode + decode in one step)
# ============================================================

@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def evaluate(pipeline, input_path, output_path, seed=0, step=25, cfg=1.0, sampler="adam2"):
    diffusion_sampler = create_sampler(sampler, step, cfg)

    image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_files.sort()
    pbar = tqdm(total=len(image_files), desc="Evaluating")

    for image_file in image_files:
        y = to_tensor(Image.open(image_file).convert("RGB")).to('cuda').unsqueeze(0)
        H, W = y.shape[-2:]
        cond, uncond = pipeline.conditioner(y)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        xT = torch.randn((1, 4, H // 8, W // 8) if pipeline.latent else (1, 3, H, W), device="cpu", dtype=torch.float32, generator=generator).to("cuda")

        sample = diffusion_sampler(pipeline.denoiser, xT, cond, uncond)
        img = pipeline.vae.decode(sample)
        img = fp2uint8(img)

        out_img = Image.fromarray(img[0].permute(1, 2, 0).cpu().numpy())
        out_path = os.path.join(output_path, os.path.basename(image_file))
        out_img.save(out_path)
        pbar.update(1)


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoD inference: compress, decompress, or evaluate")
    parser.add_argument("mode", choices=["compress", "decompress", "evaluate"])
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--input", type=str, required=True, help="Input directory (images or .cod files)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--sampler", type=str, default="adam2", choices=["adam2", "euler"])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pipeline = load_pipeline(args.config, args.ckpt)

    if args.mode == "compress":
        compress(pipeline, args.input, args.output)
    elif args.mode == "decompress":
        decompress(pipeline, args.input, args.output, seed=args.seed, step=args.step, cfg=args.cfg, sampler=args.sampler)
    elif args.mode == "evaluate":
        evaluate(pipeline, args.input, args.output, seed=args.seed, step=args.step, cfg=args.cfg, sampler=args.sampler)
