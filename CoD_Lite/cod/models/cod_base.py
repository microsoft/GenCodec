import torch
import torch.nn as nn
from cod.models.condition_codec import Codec

class CoDBase(nn.Module):
    def __init__(
            self,
            hidden_size=1152,
            ds=32,
            codebook_bits=4,
            up2x=False,
            light=False,
            *args, **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.y_embedder = Codec(ch=hidden_size, codebook_bits=codebook_bits, ds=ds, up2x=up2x, light=light)

        self.initialize_weights_base()
    
    def initialize_weights_base(self):
        # zero init final layer in codec
        self.y_embedder.decoder.initialize_weights()

    def calculate_uncond(self, uncond):
        self.y_embedder.calculate_uncond(uncond)

    def calculate_indices_bytes(self, H, W):
        return self.y_embedder.calculate_indices_bytes(H, W)

    def compress(self, x):
        return self.y_embedder.compress(x)

    def decompress(self, bitstream, H, W, device, uncond=None):
        x_hat = self.y_embedder.decompress(bitstream, H, W, device)
        if uncond is not None:
            self.calculate_uncond(uncond)
            x_hat = torch.cat([self.y_embedder.pre_calculated_uncond, x_hat], dim=0)
        return x_hat
