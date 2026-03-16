import torch
import torch.nn as nn


def uint82fp(x):
    x = x.to(torch.float32)
    x = (x - 127.5) / 127.5
    return x

def fp2uint8(x):
    x = torch.clip_((x + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)
    return x


class PixelAE(nn.Module):
    def __init__(self, scale=1.0, shift=0.0):
        super().__init__()
        self.scale = scale
        self.shift = shift

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def encode(self, x):
        return (x / self.scale + self.shift).to(torch.bfloat16)

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def decode(self, x):
        return ((x - self.shift) * self.scale).to(torch.bfloat16)


class LatentAE(nn.Module):
    def __init__(self, precompute=False, weight_path:str=None):
        super().__init__()
        self.precompute = precompute
        self.model = None
        self.weight_path = weight_path

        from diffusers.models import AutoencoderKL
        setattr(self, "model", AutoencoderKL.from_pretrained(self.weight_path))
        self.scaling_factor = self.model.config.scaling_factor

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def encode(self, x):
        assert self.model is not None
        if self.precompute:
            return x.mul_(self.scaling_factor).to(torch.bfloat16)
        return (self.model.encode(x).latent_dist.sample() * self.scaling_factor ** 2).to(torch.bfloat16)

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def decode(self, x):
        assert self.model is not None
        return self.model.decode(x / self.scaling_factor ** 2).sample.to(torch.bfloat16)


class PixelConditioner(nn.Module):
    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def __call__(self, y, metadata:dict={}):
        condition = y
        uncondition = y * 0. - 1.  # unconditional : all pixels are -1
        if condition.dtype in [torch.float64, torch.float32, torch.float16]:
            condition = condition.to(torch.bfloat16)
        if uncondition.dtype in [torch.float64, torch.float32, torch.float16]:
            uncondition = uncondition.to(torch.bfloat16)
        return condition, uncondition
