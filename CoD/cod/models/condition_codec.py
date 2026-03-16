import torch
import torch.nn as nn
import math
from cod.models.utils.vq import VectorQuantizer
from cod.models.utils.stream_utils import pack_bits, unpack_bits


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = torch.nn.Conv2d(in_channels,
                                    in_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=0)

    def forward(self, x):
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = ResnetBlock(in_channels=in_channels, out_channels=in_channels)

        nn.init.zeros_(self.block.conv2.weight)
        nn.init.zeros_(self.block.conv2.bias)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.block(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class Encoder(nn.Module):
    def __init__(self, ch, z_ch, ds, up2x, light=False):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch, kernel_size=3, stride=2 if up2x else 1, padding=1)

        # ---- Downsampling Blocks ----
        self.down_block_0 = nn.Sequential(
            ResnetBlock(in_channels=ch, out_channels=ch),
            ResnetBlock(in_channels=ch, out_channels=ch),
            Downsample(ch)
        )
        self.down_block_1 = nn.Sequential(
            ResnetBlock(in_channels=ch, out_channels=ch),
            ResnetBlock(in_channels=ch, out_channels=ch),
            Downsample(ch) if ds >= 8 else nn.Identity()
        )
        self.down_block_2 = nn.Sequential(
            ResnetBlock(in_channels=ch, out_channels=ch * 2),
            ResnetBlock(in_channels=ch * 2, out_channels=ch * 2),
            Downsample(ch * 2) if ds >= 16 else nn.Identity()
        )
        self.down_block_3 = nn.Sequential(
            ResnetBlock(in_channels=ch * 2, out_channels=ch * 2),
            ResnetBlock(in_channels=ch * 2, out_channels=ch * 2),
            Downsample(ch * 2)  if ds >= 32 else nn.Identity()
        )
        if ds <= 32:
            self.down_block_4 = nn.Sequential(
                ResnetBlock(in_channels=ch * 2, out_channels=ch * 4), AttnBlock(ch * 4),
                ResnetBlock(in_channels=ch * 4, out_channels=ch * 4), AttnBlock(ch * 4),
            )
            out_ch = ch * 4
        elif ds == 128:
            self.down_block_4 = nn.Sequential(
                ResnetBlock(in_channels=ch * 2, out_channels=ch * 4), AttnBlock(ch * 4),
                ResnetBlock(in_channels=ch * 4, out_channels=ch * 4), AttnBlock(ch * 4),
                Downsample(ch * 4),
                ResnetBlock(in_channels=ch * 4, out_channels=ch * 8), AttnBlock(ch * 8),
                ResnetBlock(in_channels=ch * 8, out_channels=ch * 8), AttnBlock(ch * 8),
                Downsample(ch * 8),
            )
            out_ch = ch * 8
        else:
            assert False

        # ---- Middle Block ----
        self.mid_block = nn.Sequential(
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch)
        )

        # ---- Final Layers ----
        self.norm_out = Normalize(out_ch)
        self.conv_out = nn.Conv2d(out_ch, z_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Initial conv
        h = self.conv_in(x)
        
        # Downsampling
        h = self.down_block_0(h)
        h = self.down_block_1(h)
        h = self.down_block_2(h)
        h = self.down_block_3(h)
        h = self.down_block_4(h)

        # Middle
        h = self.mid_block(h)

        # Final
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, out_ch, z_ch, ds, up2x, light=False):
        super().__init__()
        # ---- Initial Convolution ----
        self.conv_in = nn.Conv2d(z_ch, out_ch, kernel_size=3, stride=1, padding=1)

        # ---- Upsampling Blocks ----
        if ds <= 32:
            if not light:
                self.block = nn.Sequential(
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch),
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch),
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch),
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch)
                )
            else:
                self.block = nn.Sequential(
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch),
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch),
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch)
                )
        elif ds == 128:
            assert not light
            self.block = nn.Sequential(
                ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch),
                UpSample(out_ch),
                ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch),
                UpSample(out_ch),
                ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch),
                ResnetBlock(in_channels=out_ch, out_channels=out_ch)
            )
        else:
            assert False

        # ---- Final Layers ----
        # in: chx256x256, out: 3x256x256
        self.norm_out = Normalize(out_ch)
        self.ds = ds
        self.up2x = up2x
        if ds == 4:
            assert self.up2x
            self.conv_out = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
                ResnetBlock(in_channels=out_ch, out_channels=out_ch),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
            )
            self.ada = nn.Identity()
        elif ds == 8:
            assert self.up2x
            self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            self.ada = nn.Identity()
        elif ds == 16:
            assert self.up2x
            self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
            self.ada = nn.Identity()
        else:   # 32 or 128
            self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
            if self.up2x:
                self.ada = UpSample(out_ch)

    def initialize_weights(self):
        if isinstance(self.conv_out, nn.Sequential):
            # 对于 Sequential，初始化最后一个卷积层
            for module in self.conv_out.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
        else:
            nn.init.zeros_(self.conv_out.weight)
            nn.init.zeros_(self.conv_out.bias)

    def forward(self, z):
        # Initial conv
        h = self.conv_in(z)

        # Middle
        h = self.block(h)

        # Final
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.up2x:
            h = self.ada(h)
        return h

class Codec(nn.Module):
    def __init__(self, ch, codebook_bits, ds, up2x, light=False):
        super().__init__()
        codebook_dim = 128
        codebook_size = 2 ** codebook_bits
        self.codebook_bits = codebook_bits
        self.ds = ds
        self.up2x = up2x
        self.encoder = Encoder(128, codebook_dim, ds, up2x, light=light)
        self.bottleneck = VectorQuantizer(codebook_size, codebook_dim, 0.25)
        self.decoder = Decoder(ch, codebook_dim, ds, up2x, light=light)
        self.bottleneck.codebook_bits = codebook_bits

        self.pre_calculated_uncond = None

    def calculate_uncond(self, uncond):
        if self.pre_calculated_uncond is None:
            y = self.encoder(uncond)
            y_hat, vq_loss, min_encoding_indices = self.bottleneck(y)
            self.pre_calculated_uncond = self.decoder(y_hat)

            # self.pre_calculated_uncond = self.forward(uncond[:1])[0]

    def calculate_indices_size(self, H, W):
        real_ds = self.ds if self.up2x else self.ds // 2
        h, w = H // real_ds, W // real_ds
        return h, w

    def calculate_indices_bytes(self, H, W):
        h, w = self.calculate_indices_size(H, W)
        indices_bits = h * w * self.codebook_bits
        indices_bytes = indices_bits // 8 + int(indices_bits % 8 != 0)
        return indices_bytes
    
    def compress(self, x):
        assert x.shape[0] == 1, "Encode one image once"
        y = self.encoder(x)
        _, _, min_encoding_indices = self.bottleneck(y)
        bitstream = pack_bits(min_encoding_indices, self.codebook_bits)
        return bitstream

    def decompress(self, bitstream, H, W, device):
        h, w = self.calculate_indices_size(H, W)
        min_encoding_indices = unpack_bits(bitstream, h * w, self.codebook_bits).to(device)
        y_hat = self.bottleneck.get_codebook_entry(min_encoding_indices, (h, w))
        x_hat = self.decoder(y_hat)
        return x_hat

    def forward(self, x, fix_encoder=False):
        with torch.no_grad() if fix_encoder else torch.enable_grad():
            y = self.encoder(x)
            y_hat, vq_loss, min_encoding_indices = self.bottleneck(y)
        x_hat = self.decoder(y_hat)

        bpp_hard = torch.tensor(self.bottleneck.codebook_bits / self.ds ** 2)
        res = {
            "vq_loss": vq_loss,
            "indices": min_encoding_indices,
            "bpp_hard": bpp_hard
        }
        return x_hat, res



if __name__ == "__main__":
    import torch

    def count_all_parameters(model):
        n = sum(p.numel() for p in model.parameters())
        return f"{n / 1024 / 1024:.2f}"

    codebook_bits, ds = 8, 4       # 0.5000
    # codebook_bits, ds = 8, 8       # 0.1250
    # codebook_bits, ds = 8, 16       # 0.0313
    # codebook_bits, ds = 4, 16       # 0.0156
    # codebook_bits, ds = 8, 32       # 0.0078
    # codebook_bits, ds = 4, 32       # 0.0039    : 27 + 141 M
    # codebook_bits, ds = 4, 128     # 0.00024  : 109 + 187 M

    # model = Codec(1152, codebook_bits=codebook_bits, ds=ds, up2x=False).cuda()
    # x = torch.randn(1, 3, 256, 256).cuda()
    # x_hat, res = model(x)
    # print(res['bpp_hard'])

    model = Codec(1152, codebook_bits=codebook_bits, ds=ds, up2x=True).cuda()
    x = torch.randn(1, 3, 512, 512).cuda()
    x_hat, res = model(x)
    print(res['bpp_hard'], x.shape, x_hat.shape)

    print("Total params:", count_all_parameters(model.encoder))
    print("Total params:", count_all_parameters(model.decoder))