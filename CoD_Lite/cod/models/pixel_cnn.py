import torch
from torch import nn
import torch.nn.functional as F
import math

from cod.models.cod_base import CoDBase
from cod.models.common import RMSNorm, NerfEmbedder, NerfFinalLayer


def modulate(x, shift, scale):
    if len(x.shape) == 4:
        b, c = x.shape[:2]
        return x * (1 + scale.view(b, c, 1, 1)) + shift.view(b, c, 1, 1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim + embed_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x, cond):
        x = self.proj2(torch.cat([self.proj1(x), cond], dim=1))
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = self.in_ln(x) * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        return x + gate_mlp * h


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        patch_size,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.patch_size = patch_size

        self.cond_embed = nn.Linear(z_channels, patch_size**2*model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        x = self.input_proj(x)
        c = self.cond_embed(c)
        y = c.reshape(c.shape[0], self.patch_size**2, -1)
        for block in self.res_blocks:
            x = block(x, y)
        return x


class DiCoBlock(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1, stride=1, groups=hidden_size,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size , kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        ffn_channel = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(in_channels=hidden_size, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=hidden_size, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(hidden_size,affine=False, eps=1e-6)
        self.norm2 = LayerNorm2d(hidden_size,affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )


    def forward(self, inp, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = self.norm1(inp)

        x = modulate(x, shift_msa, scale_msa)

        x = F.gelu(self.conv2(self.conv1(x)))
        x = x * self.ca(x)
        x = self.conv3(x)

        x = inp + gate_msa.unsqueeze(-1).unsqueeze(-1) * x

        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.conv5(F.gelu(self.conv4(modulate(self.norm2(x), shift_mlp, scale_mlp))))

        return x


class DConv(CoDBase):
    """
    a Simple no-unet Convolution diffusion
    """
    def __init__(
        self,
        pred='x',
        patch_size=16,
        in_channels=3,
        hidden_size=1152,
        hidden_size_x=32,
        mlp_ratio=4.0,
        num_blocks=31,
        num_cond_blocks=28,
        bottleneck_dim=128,
        *args, **kwargs
    ):
        super().__init__(hidden_size=hidden_size, *args, **kwargs)
        self.pred = pred
        self.v_clamp_min = 0.05 if pred == 'x' else 0.0
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_cond_blocks = num_cond_blocks

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder_x = nn.Conv2d(hidden_size, hidden_size_x*patch_size**2, 1, 1, 0)

        # linear embed
        self.x_embedder = NerfEmbedder(in_channels + hidden_size_x, hidden_size_x, max_freqs=8)
        self.s_embedder = BottleneckPatchEmbed(patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # transformer
        self.blocks = nn.ModuleList([DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(self.num_cond_blocks)])
        self.dec_net = SimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=self.in_channels,  # for vlb loss
            z_channels=self.hidden_size,
            num_res_blocks=self.num_blocks-self.num_cond_blocks,
            patch_size=self.patch_size,
        )

        # linear predict
        self.final_layer = NerfFinalLayer(hidden_size_x, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.s_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.s_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, cond, return_codec_res=False, return_pred=False):
        b, c, h, w = x.shape

        input_x = x
        input_t = t

        # class and time embeddings
        t_emb = self.t_embedder(t.view(-1))
        if cond is None:
            cond, codec_res = self.y_embedder(y)
        c = t_emb

        # forward
        s = self.s_embedder(x, cond)
        for i in range(self.num_cond_blocks):
            s = self.blocks[i](s, c)

        length = s.shape[-2] * s.shape[-1]
        s = s.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)                                 # (B * length, hidden_size)

        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)  # (B, in_channels*patch_size**2, length)
        x = torch.cat([x, self.y_embedder_x(cond).flatten(2)], dim=1)                           # (B, (in_channels+hidden_size_x) * patch_size**2, length)

        x = x.reshape(b, -1, self.patch_size**2, length).permute(0, 3, 2, 1).flatten(0, 1)      # (B * length, patch_size**2, in_channels+hidden_size_x)
        x = self.x_embedder(x)

        x = self.dec_net(x, s)
        x = self.final_layer(x)                                                                 # (B * length, patch_size**2, out_channels)
        x = x.transpose(1, 2).reshape(b, length, -1)                                            # (B, length, out_channels * patch_size**2)
        output_pred = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), (h, w), kernel_size=self.patch_size, stride=self.patch_size)

        if return_pred:
            output = output_pred
        else:
            # return v
            if self.pred == 'x':
                output = (output_pred - input_x) / (1.0 - input_t.view(-1, 1, 1, 1)).clamp_min(self.v_clamp_min)
            elif self.pred == 'v':
                output = output_pred
            else:
                assert False, f'unknown pred type {self.pred}'

        if return_codec_res:
            return output, cond, codec_res
        return output, cond
