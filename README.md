# GenCodec

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<p align="center">
  <img src="assets/gencodec.png" width="90%" />
</p>

## About

**GenCodec** is a research initiative from Microsoft Research Asia pushing the boundaries of generative models for image and video compression. We aim to build the next generation of codecs that go beyond traditional rate-distortion trade-offs, targeting:

- **High Realism & Fidelity** — Visually natural reconstructions with faithful preservation of structural detail and semantic content.
- **Extreme Compression** — Meaningful reconstruction at ultra-low bitrates where conventional codecs produce unusable artifacts.
- **Practical Real-Time Coding** — Efficient encoding and decoding suitable for real-time and on-device deployment.
- **Diverse Applications** — General-purpose compression foundations that extend to more vision tasks and applications.

This repository hosts models, code, and evaluation tools for our published generative compression methods.

## Projects

- [`CoD`](CoD) — **CoD: A Diffusion Foundation Model for Image Compression** (CVPR 2026) [[arXiv]](https://arxiv.org/abs/2511.18706) [[Models]](https://huggingface.co/zhaoyangjia/CoD)
  - Compression-native diffusion model trained from scratch for image compression
  - Supports both pixel-space and latent-space diffusion under a unified architecture
  - Downstream applications: zero-shot variable-rate coding (DiffC), one-step coding, perceptual loss

- [`CoD_Lite`](CoD_Lite) — **CoD-Lite: Real-Time Diffusion-Based Generative Image Compression** [[arXiv]](https://arxiv.org/abs/2604.12525) [[Models]](https://huggingface.co/zhaoyangjia/CoD_Lite)
  - Lightweight real-time codec: 28M encoder + 52M decoder & 60 FPS encoding / 42 FPS decoding at 1080p on a single A100 GPU
  - Powered by a one-step pixel-space convolutional diffusion-based decoding framework.
  - Effective training pipeline: Compression-oriented diffusion pre-train → Distillation-guided and adversarial one-step fine-tune

## License

GenCodec is MIT licensed, as found in the [LICENSE](LICENSE) file.

## Citation

If you find this work useful, please cite the relevant project:

```bibtex
@inproceedings{jia2025cod,
    title     = {CoD: A Diffusion Foundation Model for Image Compression},
    author    = {Jia, Zhaoyang and Zheng, Zihan and Xue, Naifu and Li, Jiahao and Li, Bin and Guo, Zongyu and Zhang, Xiaoyi and Li, Houqiang and Lu, Yan},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2026}
}

@article{jia2026codlite,
    title     = {Real-Time Diffusion-Based Generative Image Compression},
    author    = {Jia, Zhaoyang and Xue, Naifu and Zheng, Zihan and Li, Jiahao and Li, Bin and Zhang, Xiaoyi and Guo, Zongyu and Zhang, Yuan and Li, Houqiang and Lu, Yan},
    journal   = {arXiv preprint arXiv:2604.12525},
    year      = {2026}
}
```
