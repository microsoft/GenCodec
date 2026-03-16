from downstream.diffc.lib.diffc.utils.alpha_beta import get_alpha_prod_and_beta_prod


def Q(noisy_latent, target_latent, current_snr, prev_snr):
    alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(current_snr)
    alpha_prod_t_prev, beta_prod_t_prev = get_alpha_prod_and_beta_prod(prev_snr)
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    mu = (
        pred_original_sample_coeff * target_latent + current_sample_coeff * noisy_latent
    )

    return mu
