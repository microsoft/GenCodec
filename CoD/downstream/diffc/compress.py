import os
import re
import torch
from PIL import Image
from pathlib import Path
import yaml
import struct
from easydict import EasyDict as edict

from downstream.diffc.lib import image_utils
from downstream.diffc.lib.diffc.encode import encode
from downstream.diffc.lib.diffc.rcc.gaussian_channel_simulator import GaussianChannelSimulator
from downstream.diffc.lib.models.CoD import CoDModel


def write_diffc_file(bitstream_cod, bitstream_diffc, width, height, step_idx, output_path):
    # Write width (2 bytes), height (2 bytes), step_idx (2 bytes), compressed image data
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<H', width))          # Write width as 2-byte little-endian uint
        f.write(struct.pack('<H', height))         # Write height as 2-byte little-endian uint
        f.write(struct.pack('<H', step_idx))       # Write step_idx as 2-byte little-endian uint
        f.write(bytes(bitstream_cod))
        f.write(bytes(bitstream_diffc))



def compress_image(image_path, output_path, noise_prediction_model, 
                  gaussian_channel_simulator, config):
    # Load and preprocess image
    img_pil = Image.open(image_path)
    img_width, img_height = img_pil.size
    gt_pt = image_utils.pil_to_torch_img(img_pil)
    gt_latent = noise_prediction_model.image_to_latent(gt_pt)
    
    # Configure model
    bitstream_cod = noise_prediction_model.configure(
        gt_pt, config.encoding_guidance_scale
    )

    # Encode image
    chunk_seeds_per_step, Dkl_per_step, _, recon_step_indices = encode(
        gt_latent,
        config.encoding_timesteps,
        noise_prediction_model,
        gaussian_channel_simulator,
        config.manual_dkl_per_step,
        [config.recon_timestep]  # Only encode for the specified timestep
    )
    
    # Get the compressed representation
    step_idx = recon_step_indices[0]  # Only one step since we specified one timestep
    bitstream_diffc = gaussian_channel_simulator.compress_chunk_seeds(
        chunk_seeds_per_step[: step_idx + 1], 
        Dkl_per_step[: step_idx + 1]
    )

    write_diffc_file(
        bitstream_cod,
        bitstream_diffc,
        img_width,
        img_height,
        step_idx,
        output_path)

def compress(diffc_cfg_path, recon_timestep, cod_cfg_path, cod_pretrained_path, image_path=None, image_dir=None, output_dir=None, dtype=torch.bfloat16):
    with open(diffc_cfg_path, "r") as f:
        config = edict(yaml.safe_load(f))
    config.recon_timestep = recon_timestep

    ###############################################################################
    ## Get image paths
    ###############################################################################

    image_paths = []

    if not bool(image_path) ^ bool(image_dir):
        raise ValueError("Must specify exactly one of --image_path or --image_dir")

    if image_path:
        image_paths.append(Path(image_path))
    else:
        image_dir = Path(image_dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        image_paths = list(map(Path, image_paths))

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ###############################################################################
    ## Make GaussianChannelSimulator and LatentNoisePredictionModel
    ###############################################################################

    gaussian_channel_simulator = GaussianChannelSimulator(
        config.max_chunk_size, config.chunk_padding
    )

    noise_prediction_model = CoDModel(cod_cfg_path, cod_pretrained_path, dtype=dtype)

    ###############################################################################
    ## Compress the provided images
    ###############################################################################

    for image_path in image_paths:
        output_path = output_dir / f"{image_path.stem}.diffc"
        compress_image(
            image_path, 
            output_path,
            noise_prediction_model, 
            gaussian_channel_simulator,
            config
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DiffC compress: image → .diffc bitstream")
    parser.add_argument("--diffc_cfg", type=str, required=True, help="DiffC config YAML")
    parser.add_argument("--cod_cfg", type=str, required=True, help="CoD model config YAML")
    parser.add_argument("--ckpt", type=str, required=True, help="Pretrained CoD checkpoint")
    parser.add_argument("--recon_timestep", type=int, default=900, help="Reconstruction timestep (controls rate)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_path", type=str, help="Single image path")
    group.add_argument("--image_dir", type=str, help="Directory of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    compress(args.diffc_cfg, args.recon_timestep, args.cod_cfg, args.ckpt,
             image_path=args.image_path, image_dir=args.image_dir, output_dir=args.output_dir)
