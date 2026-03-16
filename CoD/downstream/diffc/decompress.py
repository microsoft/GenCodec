import torch
from PIL import Image
from pathlib import Path
import yaml
import struct
from easydict import EasyDict as edict

from downstream.diffc.lib import image_utils
from downstream.diffc.lib.diffc.denoise import denoise
from downstream.diffc.lib.diffc.decode import decode
from downstream.diffc.lib.diffc.rcc.gaussian_channel_simulator import GaussianChannelSimulator
from downstream.diffc.lib.models.CoD import CoDModel


def read_diffc_file(file_path, noise_prediction_model):
    with open(file_path, 'rb') as f:
        # Read width, height, and step_idx (2 bytes each)
        width = struct.unpack('<H', f.read(2))[0]
        height = struct.unpack('<H', f.read(2))[0]
        step_idx = struct.unpack('<H', f.read(2))[0]

        # Read remaining bytes for image data
        bitstream_cod_length = noise_prediction_model.calculate_indices_bytes(height, width)
        bitstream_cod = list(f.read(bitstream_cod_length))
        bitstream_diffc = list(f.read())
    
    return width, height, step_idx, bitstream_cod, bitstream_diffc


def decompress_file(input_path, output_path, noise_prediction_model, 
                   gaussian_channel_simulator, config):
    # Read compressed data
    W, H, step_idx, bitstream_cod, bitstream_diffc = read_diffc_file(input_path, noise_prediction_model)
    
    # Decompress the representation
    chunk_seeds_per_step = gaussian_channel_simulator.decompress_chunk_seeds(
        bitstream_diffc, config.manual_dkl_per_step[:step_idx+1]
    )

    timestep = config.encoding_timesteps[step_idx]
    
    # Configure model with caption
    temp = torch.zeros((1, 3, H, W), device=noise_prediction_model.device, dtype=torch.float32)
    _, uncond = noise_prediction_model.conditioner(temp)
    noise_prediction_model.configure_decompress(
        bitstream_cod, config.encoding_guidance_scale, H, W, noise_prediction_model.device, uncond, 
    )
    
    # Get the noisy reconstruction    
    noisy_recon = decode(
        W,
        H,
        config.encoding_timesteps,
        noise_prediction_model,
        gaussian_channel_simulator,
        chunk_seeds_per_step,
        config.manual_dkl_per_step,
        seed=0)
    
    # Denoise
    recon_latent = denoise(
        noisy_recon,
        timestep,
        config.denoising_timesteps,
        noise_prediction_model
    )
    
    # Convert to image and save
    recon_img_pt = noise_prediction_model.latent_to_image(recon_latent)
    image_utils.torch_to_pil_img(recon_img_pt).save(output_path)

def decompress(diffc_cfg_path, cod_cfg_path, cod_pretrained_path, input_path=None, input_dir=None, output_dir=None, dtype=torch.bfloat16):
    with open(diffc_cfg_path, "r") as f:
        config = edict(yaml.safe_load(f))
    assert config.manual_dkl_per_step is not None, "Config must specify a manual_dkl_per_step to perform decompression."

    ###############################################################################
    ## Get input paths
    ###############################################################################

    input_paths = []

    if not bool(input_path) ^ bool(input_dir):
        raise ValueError("Must specify exactly one of --input_path or --input_dir")

    if input_path:
        input_paths.append(Path(input_path))
    else:
        input_dir = Path(input_dir)
        input_paths = list(input_dir.glob("*.diffc"))

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ###############################################################################
    ## Make GaussianChannelSimulator and LatentNoisePredictionModel
    ###############################################################################

    gaussian_channel_simulator = GaussianChannelSimulator(
        config.max_chunk_size, config.chunk_padding
    )

    noise_prediction_model = CoDModel(cod_cfg_path, cod_pretrained_path, dtype=dtype)

    # Process each file
    for input_path in input_paths:
        # Create output path: {original_name}.png
        output_path = output_dir / f"{input_path.stem}.png"
        
        decompress_file(
            input_path,
            output_path,
            noise_prediction_model,
            gaussian_channel_simulator,
            config
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DiffC decompress: .diffc bitstream → image")
    parser.add_argument("--diffc_cfg", type=str, required=True, help="DiffC config YAML")
    parser.add_argument("--cod_cfg", type=str, required=True, help="CoD model config YAML")
    parser.add_argument("--ckpt", type=str, required=True, help="Pretrained CoD checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_path", type=str, help="Single .diffc file")
    group.add_argument("--input_dir", type=str, help="Directory of .diffc files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    decompress(args.diffc_cfg, args.cod_cfg, args.ckpt,
               input_path=args.input_path, input_dir=args.input_dir, output_dir=args.output_dir)
