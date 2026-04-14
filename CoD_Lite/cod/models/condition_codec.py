import torch
import torch.nn as nn
from einops import rearrange


# ============================================================
#  Stream Utils
# ============================================================

def pack_bits(indices: torch.Tensor, bits: int) -> bytes:
    """Pack indices (1D LongTensor) into a bitstream of exact length."""
    assert indices.ndim == 1
    n = indices.numel()
    if n == 0:
        return b""
    indices = indices.to(dtype=torch.int64, device="cpu")
    total_bits = n * bits
    total_bytes = (total_bits + 7) // 8
    out = torch.zeros(total_bytes, dtype=torch.uint8)
    bit_positions = torch.arange(n, dtype=torch.int64) * bits
    for bit_idx in range(bits):
        bit_vals = (indices >> (bits - 1 - bit_idx)) & 1
        global_bit_pos = bit_positions + bit_idx
        byte_idx = global_bit_pos // 8
        bit_offset = 7 - (global_bit_pos % 8)
        out.scatter_add_(0, byte_idx, (bit_vals << bit_offset).to(torch.uint8))
    return bytes(out.numpy())


def unpack_bits(b: bytes, n: int, bits: int) -> torch.Tensor:
    """Unpack exactly n indices from bitstream b."""
    if n == 0:
        return torch.tensor([], dtype=torch.long)
    byte_tensor = torch.frombuffer(bytearray(b), dtype=torch.uint8).to(torch.int64)
    out = torch.zeros(n, dtype=torch.int64)
    bit_positions = torch.arange(n, dtype=torch.int64) * bits
    for bit_idx in range(bits):
        global_bit_pos = bit_positions + bit_idx
        byte_idx = global_bit_pos // 8
        bit_offset = 7 - (global_bit_pos % 8)
        bit_vals = (byte_tensor[byte_idx] >> bit_offset) & 1
        out |= bit_vals << (bits - 1 - bit_idx)
    return out


# ============================================================
#  Vector Quantizer
# ============================================================

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view([1, shape[0], shape[1], -1])
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


# ============================================================
#  Codec Building Blocks
# ============================================================

def nonlinearity(x):
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

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
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

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
    def __init__(self, in_channels, patch_size=16):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if not self.training:
            return self.forward_patch(x)
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)
        k = k.reshape(b,c,h*w)
        w_ = torch.bmm(q,k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)
        h_ = torch.bmm(v,w_)
        h_ = h_.reshape(b,c,h,w)
        h_ = self.proj_out(h_)
        return x+h_

    def forward_patch(self, x):
        h_ = x
        h_ = self.norm(h_)
        Q = self.q(h_)
        K = self.k(h_)
        V = self.v(h_)
        d = self.patch_size
        b, c, H, W = Q.shape
        # ds=32: feature map (16x16) is smaller than patch_size (32),
        # replicate padding would pollute attention; use global attention instead.
        if H <= d and W <= d:
            q = Q.reshape(b, c, H * W).permute(0, 2, 1)
            k = K.reshape(b, c, H * W)
            w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2).permute(0, 2, 1)
            h_ = torch.bmm(V.reshape(b, c, H * W), w_).reshape(b, c, H, W)
            return x + self.proj_out(h_)
        pad_h = (d - H % d) % d
        pad_w = (d - W % d) % d
        if pad_h > 0 or pad_w > 0:
            Q = torch.nn.functional.pad(Q, (0, pad_w, 0, pad_h), mode='replicate')
            K = torch.nn.functional.pad(K, (0, pad_w, 0, pad_h), mode='replicate')
            V = torch.nn.functional.pad(V, (0, pad_w, 0, pad_h), mode='replicate')
        _, _, H_pad, W_pad = Q.shape
        num_patches_h, num_patches_w = H_pad // d, W_pad // d
        num_patches = num_patches_h * num_patches_w
        Q = Q.reshape(b, c, num_patches_h, d, num_patches_w, d).permute(0, 2, 4, 1, 3, 5)
        K = K.reshape(b, c, num_patches_h, d, num_patches_w, d).permute(0, 2, 4, 1, 3, 5)
        V = V.reshape(b, c, num_patches_h, d, num_patches_w, d).permute(0, 2, 4, 1, 3, 5)
        Q = Q.reshape(b * num_patches, c, d * d)
        K = K.reshape(b * num_patches, c, d * d)
        V = V.reshape(b * num_patches, c, d * d)
        q = Q.permute(0, 2, 1)
        w_ = torch.bmm(q, K)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(V, w_)
        h_ = h_.reshape(b, num_patches_h, num_patches_w, c, d, d).permute(0, 3, 1, 4, 2, 5)
        h_ = h_.reshape(b, c, H_pad, W_pad)
        if pad_h > 0 or pad_w > 0:
            h_ = h_[:, :, :H, :W]
        h_ = self.proj_out(h_)
        return x + h_


# ============================================================
#  Encoder / Decoder / Codec
# ============================================================

class Encoder(nn.Module):
    def __init__(self, ch, z_ch, ds, up2x, light=False):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch, kernel_size=3, stride=2 if up2x else 1, padding=1)

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
            Downsample(ch * 2) if ds >= 32 else nn.Identity()
        )
        if ds <= 32:
            self.down_block_4 = nn.Sequential(
                ResnetBlock(in_channels=ch * 2, out_channels=ch * 4), AttnBlock(ch * 4, patch_size=32),
                ResnetBlock(in_channels=ch * 4, out_channels=ch * 4), AttnBlock(ch * 4, patch_size=32),
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

        self.mid_block = nn.Sequential(
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch)
        )
        self.norm_out = Normalize(out_ch)
        self.conv_out = nn.Conv2d(out_ch, z_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.down_block_0(h)
        h = self.down_block_1(h)
        h = self.down_block_2(h)
        h = self.down_block_3(h)
        h = self.down_block_4(h)
        h = self.mid_block(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, out_ch, z_ch, ds, up2x, light=False):
        super().__init__()
        self.conv_in = nn.Conv2d(z_ch, out_ch, kernel_size=3, stride=1, padding=1)

        if ds <= 32:
            if not light:
                self.block = nn.Sequential(
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch, patch_size=32),
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch, patch_size=32),
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch, patch_size=32),
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch)
                )
            else:
                self.block = nn.Sequential(
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch, patch_size=32),
                    ResnetBlock(in_channels=out_ch, out_channels=out_ch), AttnBlock(out_ch, patch_size=32),
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
        else:
            self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
            if self.up2x:
                self.ada = UpSample(out_ch)

    def initialize_weights(self):
        if isinstance(self.conv_out, nn.Sequential):
            for module in self.conv_out.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
        else:
            nn.init.zeros_(self.conv_out.weight)
            nn.init.zeros_(self.conv_out.bias)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.block(h)
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
