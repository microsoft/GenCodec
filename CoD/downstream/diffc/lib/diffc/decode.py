import torch
from downstream.diffc.lib.diffc.utils.p import P
from tqdm import tqdm

@torch.no_grad()
def decode(
        image_width,
        image_height,
        timestep_schedule,
        noise_prediction_model,
        gaussian_channel_simulator,
        chunk_seeds_per_step,
        Dkl_per_step,
        seed,
    ):
    """Decodes a compressed image representation back into its latent space form.

    Args:
        latent_shape (torch.Size): Shape of the target latent tensor to be reconstructed.
        timestep_schedule (List[float]): List of timesteps in decreasing order.
        predict_noise (callable): Function that predicts the noise component given a noisy
            latent and its SNR.
        gaussian_channel_simulator: Simulator used for gaussian channel reconstruction.
        chunk_seeds_per_step (List[List[int]]): Compressed representation of the image,
            consisting of lists of integer seeds for each denoising step.
        Dkl_per_step (List[float]): List of Kullback-Leibler divergence values per step,
            used to reconstruct the denoising process.
        seed (int): Random seed for reproducibility of the denoising process.

    Returns:
        torch.Tensor: The reconstructed latent representation of the image, obtained
            through progressive denoising steps guided by the compressed representation.
    """


    device = noise_prediction_model.device
    dtype = noise_prediction_model.dtype

    dummy_image = torch.zeros((1, 3, image_height, image_width)).to(device).to(dtype)
    dummy_latent = noise_prediction_model.image_to_latent(dummy_image)
    
    torch.manual_seed(seed)
    noisy_latent = torch.randn(dummy_latent.shape, device=device, dtype=dtype)

    current_timestep = 1000
    current_snr = noise_prediction_model.get_timestep_snr(current_timestep)
    for step_index, (prev_timestep, chunk_seeds, Dkl) in tqdm(enumerate(
        zip(timestep_schedule, chunk_seeds_per_step, Dkl_per_step)
    ), total=len(chunk_seeds_per_step)):
        noise_prediction = noise_prediction_model.predict_noise(
            noisy_latent, current_timestep
        )
        prev_snr = noise_prediction_model.get_timestep_snr(prev_timestep)
        p_mu, std = P(noisy_latent, noise_prediction, current_snr, prev_snr)
        sample = gaussian_channel_simulator.decode(
            chunk_seeds, noisy_latent.numel(), Dkl, seed=step_index
        )
        reshaped_sample = (
            torch.tensor(sample).reshape(noisy_latent.shape).to(device).to(dtype)
        )
        noisy_latent = reshaped_sample * std + p_mu
        current_timestep = prev_timestep
        current_snr = prev_snr

    return noisy_latent
