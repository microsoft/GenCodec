import torch

@torch.no_grad()
def copy_params(src_model, dst_model):
    for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
        dst_param.data.copy_(src_param.data)
