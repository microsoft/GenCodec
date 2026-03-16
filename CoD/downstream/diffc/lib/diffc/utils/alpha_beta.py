import torch


def get_alpha_prod_and_beta_prod(snr):
    if snr == torch.inf:
        alpha_prod = 1
    else:
        alpha_prod = snr ** 2 / (1 + snr ** 2)
    beta_prod = 1 - alpha_prod
    return alpha_prod, beta_prod
