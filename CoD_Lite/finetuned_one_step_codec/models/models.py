import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from cod.models.pixel_cnn import DConv, DiCoBlock, modulate
from cod.models.utils.encoder import DINOv2
from cod.utils.model_loader import load_pretrained_state_dict


# ============================================================
#  OneStepCoD
# ============================================================

def replace_with_cache(module, c):
    for name, child in module.named_children():
        if isinstance(child, DiCoBlock):
            cached = CacheDiCoBlock(child)
            cached.convert_cache(c)
            setattr(module, name, cached)
        else:
            replace_with_cache(child, c)


class CacheDiCoBlock(nn.Module):
    def __init__(self, block: DiCoBlock):
        super().__init__()
        self.block = block
        self.shift_msa, self.scale_msa, self.gate_msa = None, None, None
        self.shift_mlp, self.scale_mlp, self.gate_mlp = None, None, None

    @torch.no_grad()
    def convert_cache(self, c):
        self.shift_msa, self.scale_msa, self.gate_msa, \
            self.shift_mlp, self.scale_mlp, self.gate_mlp = \
            self.block.adaLN_modulation(c).chunk(6, dim=-1)
        del self.block.adaLN_modulation

    def forward(self, inp, c):
        x = self.block.norm1(inp)
        x = modulate(x, self.shift_msa, self.scale_msa)
        x = F.gelu(self.block.conv2(self.block.conv1(x)))
        x = x * self.block.ca(x)
        x = self.block.conv3(x)
        x = inp + self.gate_msa.unsqueeze(-1).unsqueeze(-1) * x
        x = x + self.gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.block.conv5(
            F.gelu(self.block.conv4(modulate(self.block.norm2(x), self.shift_mlp, self.scale_mlp))))
        return x


class OneStepCoD(DConv):
    def __init__(self, noise_level=1.0, fix_encoder=True, net_ckpt_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_level = noise_level
        self.fix_encoder = fix_encoder
        if net_ckpt_path is not None:
            self.load_pretrained(net_ckpt_path)
        self.flag_merge = False

    def eval(self):
        if not self.flag_merge:
            t = torch.zeros((1,), device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
            c = self.t_embedder(t)
            replace_with_cache(self, c)
            self.flag_merge = True
        super().eval()

    def load_pretrained(self, net_ckpt_path, pretrained_ema=True):
        assert net_ckpt_path is not None
        print(f"Loading Network weights from {net_ckpt_path} (ema=True, strict=True)")
        pretrained_dict = load_pretrained_state_dict(net_ckpt_path)
        net_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith("ema_denoiser."):
                net_dict[k.replace("ema_denoiser.", "")] = v
        self.load_state_dict(net_dict, strict=True)

    def inference(self, y, cond=None):
        b, c, h, w = y.shape
        t = torch.zeros((b), device=y.device, dtype=y.dtype)
        noise = torch.randn_like(y) * self.noise_level
        out = self.forward(noise, t, y, cond=cond, return_pred=True)[0]
        if self.pred == 'v':
            out = noise + out
        return out

    def train_step(self, y):
        b, c, h, w = y.shape
        t = torch.zeros((b), device=y.device, dtype=y.dtype)
        noise = torch.randn_like(y) * self.noise_level
        cond, codec_res = self.y_embedder(y, fix_encoder=self.fix_encoder)
        out = self.forward(noise, t, y, cond=cond, return_pred=True)[0]
        if self.pred == 'v':
            out = noise + out
        return out, codec_res


# ============================================================
#  Discriminator
# ============================================================

class SpectralConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class Discriminator(nn.Module):
    def __init__(self, indices=[2, 5, 8, 11], *args, **kwargs):
        super().__init__()
        self.backbone = DINOv2(*args, **kwargs)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.layer_indices = indices
        self.embed_dim = self.backbone.encoder.embed_dim

        self.heads = nn.ModuleList([
            nn.Sequential(
                SpectralConv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralConv2d(self.embed_dim, self.embed_dim // 2, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralConv2d(self.embed_dim // 2, 1, kernel_size=1)
            ) for _ in range(len(indices))
        ])

    def forward(self, x):
        layers_features = self.backbone.get_intermediate_layers(x, n=self.layer_indices, reshape=True)
        all_logits = []
        for feat, head in zip(layers_features, self.heads):
            logits = head(feat)
            all_logits.append(logits)
        return torch.cat(all_logits, dim=1)
