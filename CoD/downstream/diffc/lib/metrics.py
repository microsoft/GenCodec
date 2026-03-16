from downstream.diffc.lib import image_utils
import torch


def get_bpp(seed_tuples, zipf_s_vals, zipf_n_vals, recon_step_idx, num_pixels):
    from zipf_encoding import encode_zipf

    # get seeds up to and including the recon_step_idx
    recon_seeds = sum(map(list, seed_tuples[: recon_step_idx + 1]), [])

    encoding = encode_zipf(
        zipf_s_vals[: len(recon_seeds)], zipf_n_vals[: len(recon_seeds)], recon_seeds
    )

    num_bits = len(encoding) * 8

    bpp = num_bits / num_pixels
    return bpp


def get_psnr(recon, gt_pt):
    from skimage.metrics import peak_signal_noise_ratio

    return peak_signal_noise_ratio(
        image_utils.torch_to_np_img(recon), image_utils.torch_to_np_img(gt_pt)
    )


_get_lpips = None


def get_lpips(recon, gt):
    global _get_lpips
    if _get_lpips is None:
        from lpips import LPIPS

        _get_lpips = LPIPS(net="alex").to(torch.device("cuda"))
    return _get_lpips(recon * 2 - 1, gt * 2 - 1).item()


###############################################################################
## CLIP
###############################################################################


def load_clip_model(model_name="ViT-B/32", device=None):
    import clip
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device


def preprocess_image(image, preprocess):
    return preprocess(image).unsqueeze(0)


@torch.no_grad()
def clip_score(image_a, image_b, model=None, preprocess=None, device=None):
    if model is None or preprocess is None or device is None:
        model, preprocess, device = load_clip_model()

    # Preprocess images
    image_a_preprocessed = preprocess_image(image_a, preprocess).to(device)
    image_b_preprocessed = preprocess_image(image_b, preprocess).to(device)

    # Encode images
    image_a_features = model.encode_image(image_a_preprocessed)
    image_b_features = model.encode_image(image_b_preprocessed)

    # Normalize features
    image_a_features = image_a_features / image_a_features.norm(dim=1, keepdim=True)
    image_b_features = image_b_features / image_b_features.norm(dim=1, keepdim=True)

    # Calculate CLIP score
    logit_scale = model.logit_scale.exp()
    score = logit_scale * (image_a_features * image_b_features).sum()

    return score.item()


model, preprocess, device = None, None, None


def get_clip_score(recon, gt):
    global model, preprocess, device
    if model is None:
        model, preprocess, device = load_clip_model()
    return clip_score(
        image_utils.torch_to_pil_img(recon),
        image_utils.torch_to_pil_img(gt),
        model,
        preprocess,
        device,
    )
