import os
import torch


def load_pretrained_state_dict(pretrained_ckpt_path):
    # Handle split DDP checkpoints
    if os.path.exists(os.path.join(pretrained_ckpt_path, "checkpoint-state_dict.pt")):
        pretrained_ckpt_path = os.path.join(pretrained_ckpt_path, "checkpoint-state_dict.pt")
    elif os.path.isdir(pretrained_ckpt_path):
        pretrained_ckpt_path = os.path.join(pretrained_ckpt_path, "checkpoint", "mp_rank_00_model_states.pt")

    state = torch.load(pretrained_ckpt_path, map_location="cpu")
    if "module" in state:
        return state["module"]
    elif "state_dict" in state:
        return state["state_dict"]
    return state


def load_pretrained_model(model, pretrained_ckpt_path, pretrained_ema, strict=False, log=True):
    if log:
        print(f"Loading pretrained weights from {pretrained_ckpt_path} (ema={pretrained_ema}, strict={strict})")

    pretrained_dict = load_pretrained_state_dict(pretrained_ckpt_path)
    model_dict = model.state_dict()

    matched_dict = {}
    skipped_keys = []

    for k, v in pretrained_dict.items():
        if k in model_dict:
            if pretrained_ema and k.startswith("denoiser."):
                v = pretrained_dict[f"ema_{k}"]
            if model_dict[k].shape == v.shape:
                matched_dict[k] = v
            else:
                skipped_keys.append((k, f"shape mismatch: {v.shape} vs {model_dict[k].shape}"))
        else:
            skipped_keys.append((k, "key not found"))

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=strict)

    if log:
        print(f"Loaded {len(matched_dict)} parameters, skipped {len(skipped_keys)}")
        for item in skipped_keys:
            print(item)

    return model
