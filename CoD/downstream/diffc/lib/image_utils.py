import torch
import numpy as np


def np_to_torch_img(img_np, dtype=torch.float16):
    img_pt = torch.tensor(img_np.astype("float") / 255)
    img_pt = img_pt.permute(2, 0, 1).unsqueeze(0).to(dtype).to("cuda")
    return img_pt


def pil_to_torch_img(img_pil, dtype=torch.float16):
    return np_to_torch_img(np.array(img_pil), dtype=dtype)


def torch_to_np_img(img):
    return img[0].float().permute(1, 2, 0).clip(0, 1).detach().cpu().numpy()


def np_to_pil_img(img):
    from PIL import Image

    return Image.fromarray((img * 255).astype("uint8"))


def torch_to_pil_img(img):
    return np_to_pil_img(torch_to_np_img(img))
