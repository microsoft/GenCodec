import os
import re
import torch
from PIL import Image
from downstream.diffc.lib import image_utils
from downstream.diffc.lib import metrics
from downstream.diffc.lib.diffc.encode import encode
from downstream.diffc.lib.diffc.denoise import denoise
from downstream.diffc.lib.diffc.rcc.gaussian_channel_simulator import GaussianChannelSimulator
from downstream.diffc.lib.models.CoD import CoDModel
from easydict import EasyDict as edict
import yaml
from pathlib import Path
import pandas as pd


def evaluate(diffc_cfg_path, cod_cfg_path, cod_pretrained_path, image_path=None, image_dir=None, output_dir=None, dtype=torch.bfloat16):
    with open(diffc_cfg_path, "r") as file:
        config = edict(yaml.safe_load(file))

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
    ## Evaluate on the provided images
    ###############################################################################

    results_data = []

    for image_path in image_paths:

        ## Load and preprocess the image
        ###########################################################################

        img_pil = Image.open(image_path)
        img_width, img_height = img_pil.size
        gt_pt = image_utils.pil_to_torch_img(img_pil, dtype=dtype)
        gt_latent = noise_prediction_model.image_to_latent(gt_pt)
        noise_prediction_model.configure(gt_pt, config.encoding_guidance_scale)

        ## Encode the image
        ###########################################################################

        chunk_seeds_per_step, Dkl_per_step, noisy_recons, noisy_recon_step_indices = encode(
            gt_latent,
            config.encoding_timesteps,
            noise_prediction_model,
            gaussian_channel_simulator,
            config.manual_dkl_per_step,
            config.recon_timesteps,
        )

        ## Create reconstructions, save them to disk, evaluate metrics
        ###########################################################################

        noise_prediction_model.configure(gt_pt, config.denoising_guidance_scale)

        for noisy_recon, step_idx in zip(noisy_recons, noisy_recon_step_indices):

            bytes = gaussian_channel_simulator.compress_chunk_seeds(
                chunk_seeds_per_step[: step_idx + 1], Dkl_per_step[: step_idx + 1]
            )
            num_pixels = img_width * img_height
            prompt_bpp = noise_prediction_model.calc_prompt_bpp()
            bpp = len(bytes) * 8 / num_pixels + prompt_bpp

            timestep = config.encoding_timesteps[step_idx]
            snr = noise_prediction_model.get_timestep_snr(timestep).item()

            recon_latent = denoise(
                noisy_recon, timestep, config.denoising_timesteps, noise_prediction_model
            )

            recon_img_pt = noise_prediction_model.latent_to_image(recon_latent)

            psnr = metrics.get_psnr(gt_pt, recon_img_pt)
            lpips = metrics.get_lpips(gt_pt, recon_img_pt)

            timestep_dir = output_dir / str(int(timestep)).zfill(3) / "recon"
            timestep_dir.mkdir(exist_ok=True, parents=True)
            recon_path = timestep_dir / image_path.name
            image_utils.torch_to_pil_img(recon_img_pt).save(recon_path)

            results_data.append(
                {
                    "gt_path": str(image_path),
                    "recon_path": str(recon_path),
                    "recon_step_idx": step_idx,
                    "recon_timestep": timestep,
                    "snr": snr,
                    "bpp": bpp,
                    "psnr": psnr,
                    "lpips": lpips,
                }
            )
            print(results_data[-1])

    ###############################################################################
    ## Write out metrics to a csv
    ###############################################################################

    results_df = pd.DataFrame(data=results_data)
    results_df.to_csv(output_dir / "results.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DiffC evaluate: compress + decompress + metrics")
    parser.add_argument("--diffc_cfg", type=str, required=True, help="DiffC config YAML")
    parser.add_argument("--cod_cfg", type=str, required=True, help="CoD model config YAML")
    parser.add_argument("--ckpt", type=str, required=True, help="Pretrained CoD checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_path", type=str, help="Single image path")
    group.add_argument("--image_dir", type=str, help="Directory of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    evaluate(args.diffc_cfg, args.cod_cfg, args.ckpt,
             image_path=args.image_path, image_dir=args.image_dir, output_dir=args.output_dir)
