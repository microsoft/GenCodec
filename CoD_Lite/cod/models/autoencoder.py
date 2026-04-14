import torch
import torch.nn as nn


def fp2uint8(x):
    x = ((x + 1) * 127.5 + 0.5).clamp_(0, 255).to(torch.uint8)
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


class PixelConditioner(nn.Module):
    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def __call__(self, y, metadata=None):
        condition = y
        uncondition = y * 0. - 1.  # unconditional : all pixels are -1
        if condition.dtype in [torch.float64, torch.float32, torch.float16]:
            condition = condition.to(torch.bfloat16)
        if uncondition.dtype in [torch.float64, torch.float32, torch.float16]:
            uncondition = uncondition.to(torch.bfloat16)
        return condition, uncondition
