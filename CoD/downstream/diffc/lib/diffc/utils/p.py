from downstream.diffc.lib.diffc.utils.alpha_beta import get_alpha_prod_and_beta_prod
import torch


def P(noisy_latent, noise_prediction, current_snr, prev_snr):
    alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(current_snr)
    alpha_prod_t_prev, beta_prod_t_prev = get_alpha_prod_and_beta_prod(prev_snr)
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    pred_original_sample = (
        noisy_latent - beta_prod_t ** (0.5) * noise_prediction
    ) / alpha_prod_t ** (0.5)

    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = (
        pred_original_sample_coeff * pred_original_sample
        + current_sample_coeff * noisy_latent
    )

    # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
    std = variance ** (0.5)
    return pred_prev_sample, std
