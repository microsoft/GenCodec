<h2 align="center">CoD: A Diffusion Foundation Model for Image Compression</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2511.18706"><img src="https://img.shields.io/badge/arXiv-2511.18706-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/jzyustc/CoD"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue.svg" alt="Hugging Face"></a>
</p>

---

<p align="center">
  <img src="assets/foundation.png" width="90%" />
</p>

## 📖 Introduction

**CoD** (**Co**mpression-oriented **D**iffusion) is the first diffusion foundation model designed and trained from scratch specifically for image compression. Unlike prior approaches that repurpose pretrained text-to-image models like Stable Diffusion, CoD is trained end-to-end on image-only data with a native encoder–decoder architecture tailored for compression: a condition encoder extract iamge-native features, a VQ information bottleneck compresses them into a compact bitstream, and a Diffusion Transformer reconstructs the image conditioned on the quantized representation.

<p align="center">
  <img src="assets/framework.png" width="85%" />
</p>

### ✨ Key Features

🧱 **CoD as a Compression-oriented Foundation Model**
- **Compression-native design** — trained from scratch for compression, not adapted from text-to-image models
- **Training efficiency** — ~300× less training compute than Stable Diffusion (~20 A100 GPU days), using only fully-open image datasets
- **Towrads pixel diffusion** — supporting **pixel** diffusion and **latent** diffusion models under one unified architecture
- **Towrads extremely-low bitrates** — from 0.0039 bpp to extremely-low 0.00024 bpp (64 bits for 512x512 image) compression

🔌 **CoD for Downstream Applications**
- **Zero-shot variable-rate coding** — continuous rate control via [DiffC](downstream/diffc/README_DiffC.md) without retraining
- **Finetuned one-step coding** — distill into [One-step CoD](downstream/finetuned_one_step_cod.py) via DMD, demonstraing competitive performance with SOTA one-step diffusion codecs.
- **As a perceptual loss** — provide DMD-based perceptual supervision to boost other reconstruction tasks (e.g., improving FID from 92.5 to 80.9 on [MS-ILLM](downstream/perceptual_loss_inference.py)).



## 📋 Release Plans

- [x] ~~Testing code~~
- [x] ~~Training code~~
- [x] ~~Pre-trained checkpoints~~
- [x] ~~Downstream: DiffC~~
- [x] ~~Downstream: One-Step CoD~~
- [x] ~~Downstream: CoD as Perceptual Loss for MS=ILLM~~
- [ ] Training data (coming soon)

## 🏆 Performance

Metrics evaluated on Kodak (512×512):

| Model | BPP | PSNR ↑ | LPIPS ↓ | DISTS ↓ | FID ↓ | Checkpoint | Config |
|:------|:---:|:------:|:-------:|:-------:|:-----:|:----------:|:------:|
| CoD (pixel) | 0.0039 | 16.21 | 0.434 | 0.186 | 46.0 | [Download](https://huggingface.co/jzyustc/CoD/resolve/main/cod/CoD_pixel_vpred.pt) | [Config](https://huggingface.co/jzyustc/CoD/resolve/main/cod/CoD_pixel_vpred.yaml) |
| CoD (latent) | 0.0039 | 15.03 | 0.415 | 0.188 | 45.7 | [Download](https://huggingface.co/jzyustc/CoD/resolve/main/cod/CoD_latent_vpred.pt) | [Config](https://huggingface.co/jzyustc/CoD/resolve/main/cod/CoD_latent_vpred.yaml) |
| CoD (latent) | 0.00024 | 10.09 | 0.686 | 0.288 | 69.5 | [Download](https://huggingface.co/jzyustc/CoD/resolve/main/cod/CoD_latent_vpred_64bits.pt) | [Config](https://huggingface.co/jzyustc/CoD/resolve/main/cod/CoD_latent_vpred_64bits.yaml) |

All checkpoints are available on [HuggingFace](https://huggingface.co/jzyustc/CoD) (including one-step and perceptual loss models).

### Rate-Distortion Curves (DiffC + CoD)

<details>
<summary>Click to expand</summary>

<p align="center">
  <img src="assets/rd_pixel.png" width="90%" />
</p>
<p align="center">
  <img src="assets/rd_latent.png" width="90%" />
</p>

</details>

### Visual Comparison (DiffC + CoD)

<p align="center">
  <img src="assets/visual_comapre.png" width="90%" />
</p>

## 🔧 Installation

### Setup

```bash
git clone https://github.com/microsoft/jzyustc/CoD.git
cd CoD
conda create -n cod python=3.12
conda activate cod
pip install -r requirements.txt
```

### Download Checkpoints

```bash
# Download base CoD models
huggingface-cli download jzyustc/CoD --include "cod/*" --local-dir ./pretrained/CoD

# Download one-step models
huggingface-cli download jzyustc/CoD --include "finetuned_one_step_cod/*" --local-dir ./pretrained/CoD

# Download perceptual loss models
huggingface-cli download jzyustc/CoD --include "perceptual_loss_illm_dec/*" --local-dir ./pretrained/CoD

# Download everything
huggingface-cli download jzyustc/CoD --local-dir ./pretrained/CoD
```

### DINOv2 (required for training)

```bash
# Clone DINOv2 repo
git clone --depth 1 https://github.com/facebookresearch/dinov2.git \
    pretrained/torch_hub/facebookresearch_dinov2_main

# Download ViT-B/14 weights
mkdir -p pretrained/torch_hub/checkpoints
wget -O pretrained/torch_hub/checkpoints/dinov2_vitb14_pretrain.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
```

## 🚀 Inference

The inference CLI has three subcommands: `compress`, `decompress`, and `evaluate`.

### Compress

Encode images into `.cod` bitstreams:

```bash
python -m cod.inference compress \
    --ckpt ./pretrained/CoD/cod/CoD_pixel_vpred.pt \
    --config ./pretrained/CoD/cod/CoD_pixel_vpred.yaml \
    --input <image_dir> \
    --output <bitstream_dir>
```

### Decompress

Decode `.cod` bitstreams back to images:

```bash
python -m cod.inference decompress \
    --ckpt ./pretrained/CoD/cod/CoD_pixel_vpred.pt \
    --config ./pretrained/CoD/cod/CoD_pixel_vpred.yaml \
    --input <bitstream_dir> \
    --output <recon_dir> \
    --step 25 --cfg 3.0 --sampler adam2
```

### End-to-End Evaluate

Compress and decompress in a single pass:

```bash
python -m cod.inference evaluate \
    --ckpt ./pretrained/CoD/cod/CoD_pixel_vpred.pt \
    --config ./pretrained/CoD/cod/CoD_pixel_vpred.yaml \
    --input <image_dir> \
    --output <recon_dir> \
    --step 25 --cfg 3.0 --sampler adam2
```

> **Note:** CoD (latent) uses `--cfg 1.25`. CoD (pixel) and CoD (latent, 64-bit) uses `--cfg 3.0`.


### Arguments

| Argument | Type | Default | Description |
|:---------|:----:|:-------:|:------------|
| `mode` | str | — | `compress`, `decompress`, or `evaluate` |
| `--ckpt` | str | — | Path to model checkpoint |
| `--config` | str | — | Path to YAML config file |
| `--input` | str | — | Input directory (images or `.cod` files) |
| `--output` | str | — | Output directory |
| `--seed` | int | `0` | Random seed |
| `--step` | int | `25` | Diffusion sampling steps |
| `--cfg` | float | `3.0` | Classifier-free guidance scale (`3.0` for pixel / latent-64bits, `1.25` for latent) |
| `--sampler` | str | `adam2` | Sampler type (`adam2` or `euler`) |

## 🏋️ Training

### Data Preparation

> **Note:** Training data is currently being organized and will be released soon.

We train on open image datasets including [OpenImages](https://storage.googleapis.com/openimages/web/index.html), [SAM-1B](https://ai.meta.com/datasets/segment-anything/), and [ImageNet-21K](https://www.image-net.org/). Images are resized (short edge) to 256 and 512 for the two training stages. Organize the data directory as follows:

```
<data_dir>/
├── anno_256/
│   └── anno.txt                    # annotation for 256×256 training
├── anno_512/
│   └── hq_anno.txt                 # annotation for 512×512 training
├── images/
│   ├── OpenImage/
│   │   ├── train256_0/             # short-edge resized to 256 (subset 0)
│   │   │   ├── 02a0b26f3e94e09f.jpg
│   │   │   └── ...
│   │   ├── train512_0/             # short-edge resized to 512 (subset 0)
│   │   │   ├── 02a0b26f3e94e09f.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── SAM-1B/
│   │   └── ...
│   └── ImageNet-21K/
│       └── ...
└── test/
    └── kodak512/images/            # Kodak (center-cropped to 512×512)
    └── clis512/images/             # CLIC2020 (short-edge resized to 512, then center-cropped to 512×512)
    └── div512/images/              # DIV2K (short-edge resized to 512, then center-cropped to 512×512)
```

Each annotation file lists one relative path per line (relative to `<data_dir>/images/`):
```
OpenImage/train256_0/02a0b26f3e94e09f.jpg
SAM-1B/...
...
```


### Training

The full pipeline has three stages with automatic checkpoint chaining:

```bash
python entry_train_cod.py \
    --data_dir <data_dir> \
    --save_dir <save_dir> \
    --exp_name my_experiment \
    --bpp 0_0039
```

Add `--latent` for latent-space mode. Add `--dinov2_path <path>` to override the DINOv2 weight path.

| Stage | Resolution | Batch Size | Max Steps | Description |
|:-----:|:---------:|:----------:|:---------:|:------------|
| 1 | 256×256 | 128 | 400K | Low-resolution pre-train |
| 2 | 512×512 | 64 | 100K | High-resolution fine-tune|
| 3 | 512×512 | 64 | 50K | Unified post-training |

> The launcher auto-detects GPUs and adjusts per-GPU batch size with gradient accumulation.


## 📊 Evaluation

Compute image quality metrics between reference and reconstructed images:

```bash
python scripts/metric.py \
    --ref <reference_dir> \
    --recon <reconstruction_dir> \
    --device cuda:0 \
    --fid_patch_size 256 \
    --output_path <output_dir> \
    --output_name my_experiment
```

**Supported metrics:** PSNR, LPIPS (AlexNet), DISTS, FID.

> FID uses patch-based computation ([Mentzer et al., 2020](https://arxiv.org/abs/2006.09965)) for stable evaluation on small high-resolution datasets.

## 🔀 Downstream Applications

As a compression foundation model, CoD naturally supports various downstream coding algorithms.

### 1. Zero-Shot Variable-Rate Coding (DiffC)

<details>
<summary><b>Click to Expand</b></summary>

**DiffC** treats the reverse diffusion process as a communication channel. By controlling the encoding depth, DiffC enables continuous rate adjustment from ultra-low to high bitrates — without retraining the base model. See [`downstream/diffc/README_DiffC.md`](downstream/diffc/README_DiffC.md) for details.

</details>

### 2. Finetuned One-Step Coding

<details>
<summary><b>Click to Expand</b></summary>

CoD can be distilled into a real-time one-step generator via Distribution Matching Distillation (DMD).

Rate-Distortion Curves

<p align="center">
  <img src="assets/rd_one_step.png" width="90%" />
</p>


**Available models:** `bpp_0_0039_noise_1` (non-deterministic decoding), `bpp_0_0039`, `bpp_0_0312`, `bpp_0_1250` (deterministic decoding)

```bash
# Download one-step models
huggingface-cli download jzyustc/CoD --include "finetuned_one_step_cod/*" --local-dir ./pretrained/CoD
```

```bash
# Evaluate (compress + decompress in one pass)
python -m downstream.finetuned_one_step_cod evaluate \
    --ckpt ./pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.pt \
    --config ./pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.yaml \
    --input <image_dir> --output <recon_dir>

# Compress only
python -m downstream.finetuned_one_step_cod compress \
    --ckpt ./pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.pt \
    --config ./pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.yaml \
    --input <image_dir> --output <bitstream_dir>

# Decompress only
python -m downstream.finetuned_one_step_cod decompress \
    --ckpt ./pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.pt \
    --config ./pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.yaml \
    --input <bitstream_dir> --output <recon_dir>
```

</details>

### 3. CoD as a Perceptual Loss

<details>
<summary><b>Click to Expand</b></summary>

CoD provides DMD-based perceptual supervision that can boost other reconstruction tasks. For example, finetuning MS-ILLM's decoder with CoD-based DMD loss at 0.011 bpp:

| Method | PSNR ↑ | LPIPS ↓ | DISTS ↓ | FID ↓ |
|:-------|:------:|:-------:|:-------:|:-----:|
| MS-ILLM | 21.43 | 0.403 | 0.271 | 92.5 |
| + CoD DMD loss | 21.08 | 0.376 | 0.248 | 80.9 |

And more rate-distortion curves:

<p align="center">
  <img src="assets/rd_perceptual.png" width="90%" />
</p>

Visual comparison: 
<p align="center">
  <img src="assets/visual_perceptual.png" width="50%" />
</p>

**Available checkpoints:** `msillm_quality_1`, `msillm_quality_2`, `msillm_quality_3`, `msillm_quality_4`, `msillm_quality_vlo2`

Requires [NeuralCompression](https://github.com/facebookresearch/NeuralCompression) (installed automatically via `torch.hub`).

```bash
# Download perceptual loss models
huggingface-cli download jzyustc/CoD --include "perceptual_loss_illm_dec/*" --local-dir ./pretrained/CoD
```

```bash
python -m downstream.perceptual_loss_inference \
    --ckpt ./pretrained/CoD/perceptual_loss_illm_dec/msillm_quality_1.pt \
    --quality 1 \
    --input <image_dir> --output <recon_dir>
```

</details>

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{jia2025cod,
    title     = {CoD: A Diffusion Foundation Model for Image Compression},
    author    = {Jia, Zhaoyang and Zheng, Zihan and Xue, Naifu and Li, Jiahao and Li, Bin and Guo, Zongyu and Zhang, Xiaoyi and Li, Houqiang and Lu, Yan},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2026}
}
```

## 🙏 Acknowledgements

This project builds on the following excellent open-source works:

- [PixNerD](https://github.com/MCG-NJU/PixNerd) and [DDT](https://github.com/MCG-NJU/DDT) — Codebase foundation
- [DMD](https://github.com/tianweiy/DMD2) — Distribution Matching Distillation reference
- [NeuralCompression](https://github.com/facebookresearch/NeuralCompression) — MS-ILLM codec for perceptual loss experiments
- [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-ema) — Latent-space encoder/decoder