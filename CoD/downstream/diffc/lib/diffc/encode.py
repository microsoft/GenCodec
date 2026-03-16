import torch
from downstream.diffc.lib.diffc.utils.p import P
from downstream.diffc.lib.diffc.utils.q import Q
from tqdm import tqdm


@torch.no_grad()
def encode(
    target_latent,
    timestep_schedule,
    noise_prediction_model,
    gaussian_channel_simulator,
    manual_dkl_per_step=None,
    recon_timesteps=[],
    seed=0,
):
    """Creates a compressed representation of an image using a diffusion model.

    Args:
        target_latent: Latent representation of the image to encode, as produced by the
            diffusion model's VAE encoder.
        timestep_schedule: List of timesteps, parallel to SNR_schedule. The timesteps should match the SNRs that the diffusion model expects at those timesteps.
        SNR_schedule: List of signal-to-noise ratios, decreasing towards zero (e.g.,
            [0.8, 0.6, 0.4, 0.2, 0.1]). SNR values must be in the set of values expected
            by the predict_noise function. Last element must be > 0. Ending with '0'
            (lossless compression of the latent) is not currently supported (and probably
            not desirable).
        predict_noise: Callable which takes in a noisy latent and that latent's SNR, and
            returns a prediction of the latent's noise component.
        gaussian_channel_simulator: Used for gaussian channel simulation.
        manual_dkl_per_step: Used to manually hard-code the dkl per step. Otherwise we'd
            need to send it as side-information. TODO: fancier entropy models of dkl per
            step?
        recon_timesteps: List of timesteps in decreasing order. When used, saves the noisy
            latents from the encoding process at each timestep.
        seed:
            random seed for the compression process.

    Returns:
        tuple:
            - chunk_seeds_per_step (List[List[int]]): One list of ints per step. This is
              the compressed representation of the image, although it still needs to be
              entropy coded. Fed back into the gaussian channel simulator for decoding.
            - dkl_per_step (List[float]): This is also fed back in to the gaussian
              channel simulator to reconstruct the denoising process.
            - noisy_recons: Noisy reconstructions of the target image generated during
              the encoding process. These will be the same noisy reconstructions
              generated during decoding. For faster evaluation, we can skip decoding and
              just use these recons.
            - noisy_recon_step_indices (List[float]): List which is parallel to
              noisy_recons, and reports the step index for each recon.
    """
    chunk_seeds_per_step = []
    dkl_per_step = []
    noisy_recons = []
    noisy_recon_step_indices = []
    recon_timesteps = recon_timesteps.copy()

    torch.manual_seed(seed)
    noisy_latent = torch.randn(
        target_latent.shape, device=target_latent.device, dtype=target_latent.dtype
    )

    current_timestep = 1000
    current_snr = noise_prediction_model.get_timestep_snr(current_timestep)

    for step_index, prev_timestep in tqdm(
        enumerate(timestep_schedule), total=len(timestep_schedule)
    ):  # "previous" as in closer to 1 than the current snr
        noise_prediction = noise_prediction_model.predict_noise(
            noisy_latent, current_timestep
        )
        prev_snr = noise_prediction_model.get_timestep_snr(prev_timestep)
        p_mu, std = P(noisy_latent, noise_prediction, current_snr, prev_snr)
        q_mu = Q(noisy_latent, target_latent, current_snr, prev_snr)
        p_mu, std, q_mu = p_mu.half(), std.half(), q_mu.half()
        q_mu_flat_normed = ((q_mu - p_mu) / std).flatten().detach().cpu().numpy()

        manual_dkl = (
            None if manual_dkl_per_step is None else manual_dkl_per_step[step_index]
        )

        sample, chunk_seeds, dkl = gaussian_channel_simulator.encode(
            q_mu_flat_normed, manual_dkl=manual_dkl, seed=step_index
        )
        chunk_seeds_per_step.append(chunk_seeds)
        dkl_per_step.append(dkl)
        sample = torch.tensor(sample)
        reshaped_sample = (
            sample.reshape(noisy_latent.shape)
            .to(noisy_latent.device)
            .to(noisy_latent.dtype)
        )
        noisy_latent = reshaped_sample * std + p_mu
        current_timestep = prev_timestep
        current_snr = prev_snr

        ## Optionally, save the current reconstruction
        save_current_latent = False
        while len(recon_timesteps) > 0 and current_timestep <= recon_timesteps[0]:
            save_current_latent = True
            recon_timesteps = recon_timesteps[1:]

        if save_current_latent:
            noisy_recons.append(noisy_latent)
            noisy_recon_step_indices.append(step_index)

    return chunk_seeds_per_step, dkl_per_step, noisy_recons, noisy_recon_step_indices
