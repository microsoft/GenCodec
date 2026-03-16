import copy
import os
import torch
import torch.nn as nn
import timm
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class DINOv2(nn.Module):
    def __init__(self, weight_path:str, base_patch_size=16):
        super().__init__()
        directory = os.path.dirname(weight_path)
        weight_path = os.path.basename(weight_path)
        self.encoder = torch.hub.load(
            directory,
            weight_path,
            source="local",
            skip_validation=True
        )
        self.encoder = self.encoder.to(torch.bfloat16)
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()
        self.base_patch_size = base_patch_size
        self.encoder.compile()

    def get_intermediate_layers(self, x, n=[2, 5, 8, 11], reshape=True):
            b, c, h, w = x.shape
            x = torch.nn.functional.interpolate(
                x, (int(14*h/self.base_patch_size), int(14*w/self.base_patch_size)),
                mode='bicubic', align_corners=False
            )
            outputs = self.encoder.get_intermediate_layers(x, n=n, return_class_token=False)

            if reshape:
                target_h, target_w = x.shape[-2] // 14, x.shape[-1] // 14
                reshaped_outputs = []
                for feat in outputs:
                    feat = feat.reshape(b, target_h, target_w, -1).permute(0, 3, 1, 2).contiguous()
                    reshaped_outputs.append(feat.to(torch.bfloat16))
                return reshaped_outputs

            return outputs

    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, x, resize=True):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        if resize:
            x = torch.nn.functional.interpolate(x, (int(14*h/self.base_patch_size), int(14*w/self.base_patch_size)), mode='bicubic')
        feature = self.encoder.forward_features(x)['x_norm_patchtokens']
        feature = feature.to(torch.bfloat16)
        return feature
