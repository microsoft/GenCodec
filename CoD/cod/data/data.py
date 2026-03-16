import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize
from functools import partial


#########
# Train #
#########

def center_crop_fn(image, height, width):
    crop_x = (image.width - width) // 2
    crop_y = (image.height - height) // 2
    return image.crop((crop_x, crop_y, crop_x + width, crop_y + height))

class ImageDataset(Dataset):
    def __init__(self, root, metadata, resolution=256, random_crop=False):
        with open(metadata, 'r') as f:
            self.image_paths = [os.path.join(root, line.strip()) for line in f if line.strip()]
        if random_crop:
            import torchvision.transforms
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(resolution),
                torchvision.transforms.RandomCrop(resolution),
                torchvision.transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = partial(center_crop_fn, height=resolution, width=resolution)
        self.normalize = Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        raw_image = Image.open(img_path).convert('RGB')
        raw_image = self.transform(raw_image)
        raw_image = to_tensor(raw_image)
        normalized_image = self.normalize(raw_image)
        metadata = {"raw_image": raw_image}
        return normalized_image, raw_image, metadata


#########
# Test #
#########

def save_fn(image, metadata, root_path):
    if torch.is_tensor(image):
        image = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(np.uint8(image))
    image_path = os.path.join(root_path, metadata['filename'])
    image.save(image_path)


class ImageTestDataset(Dataset):
    def __init__(self, latent_shape=(3, 256, 256), root=None):
        self.image_paths = os.listdir(root)
        self.normalize = Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        self.latent_shape = latent_shape
        self.root = root

    def load_image(self, image_path):
        raw_image = Image.open(image_path).convert('RGB')
        raw_image = to_tensor(raw_image)
        return raw_image

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.root, image_name)
        raw_image = self.load_image(image_path)

        generator = torch.Generator().manual_seed(0)
        latent = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)

        metadata = dict(
            filename=image_name,
            seed=0,
            condition=raw_image,
            save_fn=save_fn,
        )
        return latent, raw_image, metadata

    def __len__(self):
        return len(self.image_paths)
