from tqdm import tqdm
from downstream.diffc.lib.diffc.utils.alpha_beta import get_alpha_prod_and_beta_prod
import torch

@torch.no_grad()
def denoise(noisy_latent, latent_timestep, timestep_schedule, noise_prediction_model):
    """
    Perform probability-flow-based denoising upon the noisy latent.

    Args:
        noisy_latent: latent to be denoised.
        latent_SNR: signal to noise ratio of the latent to be denoised.
        SNR_schedule (List[float]): List of signal-to-noise ratios in decreasing order,
            matching the schedule used during encoding. Last element should be 0 for fully denoised image.
        predict_noise (callable): Function that predicts the noise component given a noisy
            latent and its SNR.

    """
    latent = noisy_latent
    current_timestep = latent_timestep
    current_snr = noise_prediction_model.get_timestep_snr(current_timestep)

    timestep_schedule = [t for t in timestep_schedule if t < latent_timestep]

    for prev_timestep in tqdm(
        timestep_schedule
    ):  # "previous" as in higher than the current snr
        noise_prediction = noise_prediction_model.predict_noise(
            latent.to(noise_prediction_model.dtype), current_timestep
        ).to(torch.float32)
        prev_snr = noise_prediction_model.get_timestep_snr(prev_timestep)

        alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(current_snr)
        alpha_prod_t_prev, beta_prod_t_prev = get_alpha_prod_and_beta_prod(prev_snr)

        # if int(prev_timestep) == 0:
        #    from IPython.core.debugger import set_trace
        #    set_trace()

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        sample = latent
        model_output = noise_prediction
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        latent = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        current_timestep = prev_timestep
        current_snr = prev_snr

    return latent.to(noisy_latent.dtype)
