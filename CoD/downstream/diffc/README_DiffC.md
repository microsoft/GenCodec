# DiffC: Zero-Shot Downstream Coding via Diffusion

DiffC enables zero-shot variable-rate image compression using a pretrained CoD model. It leverages the reverse diffusion process as a communication channel: encoding compresses an image into a compact bitstream at any target rate, and decoding reconstructs the image by running the reverse diffusion conditioned on that bitstream.

## Installation

In addition to the CoD base environment, install the DiffC dependencies:

```bash
pip install -r downstream/diffc/requirements_diffc.txt
pip install -e downstream/diffc/lib/diffc/rcc/arithmetic-coding/python-bindings
```

> The arithmetic coding library requires a Rust toolchain. Install via [rustup](https://rustup.rs/) if not already available.

## Usage

All commands run from the CoD project root.

### Evaluate (end-to-end)

Compress + decompress + compute metrics in one pass:

```bash
python -m downstream.diffc.evaluate \
    --diffc_cfg downstream/diffc/configs/CoD-pixel-cond.yaml \
    --cod_cfg cod/configs/pixel/bpp_0_0039_pix512_xl.yaml \
    --ckpt <checkpoint.ckpt> \
    --image_dir <image_dir> \
    --output_dir <output_dir>
```

Output: reconstructed images at each reconstruction timestep + `results.csv` with per-image BPP, PSNR, and LPIPS.

### Compress

Compress images to `.diffc` bitstreams:

```bash
python -m downstream.diffc.compress \
    --diffc_cfg downstream/diffc/configs/CoD-pixel-cond.yaml \
    --cod_cfg cod/configs/pixel/bpp_0_0039_pix512_xl.yaml \
    --ckpt <checkpoint.ckpt> \
    --recon_timestep 900 \
    --image_dir <image_dir> \
    --output_dir <output_dir>
```

`--recon_timestep` controls the target rate: lower timestep = higher bitrate = better quality.

### Decompress

Decompress `.diffc` bitstreams back to images:

```bash
python -m downstream.diffc.decompress \
    --diffc_cfg downstream/diffc/configs/CoD-pixel-cond.yaml \
    --cod_cfg cod/configs/pixel/bpp_0_0039_pix512_xl.yaml \
    --ckpt <checkpoint.ckpt> \
    --input_dir <bitstream_dir> \
    --output_dir <output_dir>
```

## Arguments

**Common arguments** (all three commands):

| Argument | Description |
|----------|-------------|
| `--diffc_cfg` | DiffC config YAML |
| `--cod_cfg` | CoD model config YAML |
| `--ckpt` | Pretrained CoD checkpoint |
| `--output_dir` | Output directory |

**compress** and **evaluate**:

| Argument | Description |
|----------|-------------|
| `--image_path` | Single image (mutually exclusive with `--image_dir`) |
| `--image_dir` | Directory of images |

**compress** only:

| Argument | Default | Description |
|----------|---------|-------------|
| `--recon_timestep` | 900 | Reconstruction timestep (controls target rate) |

**decompress** only:

| Argument | Description |
|----------|-------------|
| `--input_path` | Single `.diffc` file (mutually exclusive with `--input_dir`) |
| `--input_dir` | Directory of `.diffc` files |

## Configurations

| Config | Space | Guidance | Description |
|--------|-------|----------|-------------|
| `CoD-pixel-cond.yaml` | Pixel | enc=1.0, dec=3.0 | Pixel-space with classifier-free guidance |
| `CoD-pixel-base.yaml` | Pixel | none | Pixel-space unconditional |
| `CoD-latent-cond.yaml` | Latent | enc=1.0, dec=1.25 | Latent-space with classifier-free guidance |
| `CoD-latent-base.yaml` | Latent | none | Latent-space unconditional |

## Rate Control

| Timestep | ~BPP | ~PSNR |
|----------|------|-------|
| 900 | 0.005 | 20 dB |
| 500 | 0.020 | 28 dB |
| 200 | 0.090 | 34 dB |
| 100 | 0.220 | 37 dB |
| 50 | 0.500 | 40 dB |
| 20 | 1.470 | 44 dB |

Higher timestep = fewer encoding steps = lower BPP. Lower timestep = more steps = higher BPP and quality.
