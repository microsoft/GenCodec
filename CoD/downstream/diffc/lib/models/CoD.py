import torch
import numpy as np
from omegaconf import OmegaConf
from downstream.diffc.lib.diffc.utils.alpha_beta import get_alpha_prod_and_beta_prod
from cod.utils.test_utils import instantiate_class, load_model


def sigma_to_snr(sigma):
    return (1 - sigma) / sigma


def get_ot_flow_to_ddpm_factor(snr):
    OT_flow_noise_sigma = 1 / (snr + 1)

    alpha_cumprod = snr ** 2 / (snr ** 2 + 1)
    DDPM_noise_sigma = torch.sqrt(1 - alpha_cumprod)

    ot_flow_to_ddpm_factor = DDPM_noise_sigma / OT_flow_noise_sigma

    return ot_flow_to_ddpm_factor


class CoDModel:
    def __init__(
        self,
        cfg_path,
        pretrained_path, 
        device="cuda",
        dtype=torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype

        config = OmegaConf.load(cfg_path)
        self.vae = instantiate_class(config.model.vae).to(self.device).to(self.dtype)
        self.denoiser = instantiate_class(config.model.denoiser).to(self.device).to(self.dtype)
        self.conditioner = instantiate_class(config.model.conditioner).to(self.device).to(self.dtype)

        # load pretrained weights
        ckpt = torch.load(pretrained_path, map_location="cpu")
        self.denoiser = load_model(ckpt, self.denoiser, prefix="ema_denoiser.")
        self.denoiser.eval()

        # Pre-compute SNR values for timesteps
        sigmas = np.arange(1000) / 1000 # Default 1000 timesteps
        self.snr_values = torch.tensor(
            [sigma_to_snr(sigma) for sigma in sigmas], device=device
        )

        # Initialize configuration attributes
        self.cond = None
        self.uncond = None
        self.guidance_scale = None

    def calc_prompt_bpp(self):
        return torch.tensor(self.denoiser.y_embedder.bottleneck.codebook_bits / self.denoiser.y_embedder.ds ** 2)

    def get_timestep_snr(self, timestep):
        """Return the SNR value for a given timestep."""
        if timestep == 0:
            return torch.inf
        return self.snr_values[timestep - 1]

    def image_to_latent(self, img_pt):
        """
        Convert input image tensor to latent representation.
        """
        if img_pt.dim() == 3:
            img_pt = img_pt.unsqueeze(0)

        # Move input to correct device and type
        img_pt = img_pt.to(dtype=self.dtype)
        return self.vae.encode(2 * img_pt - 1)

    def latent_to_image(self, latent):
        """Convert packed latent representation back to image."""
        return (self.vae.decode(latent) / 2 + 0.5).clamp(0, 1).detach()

    def calculate_indices_bytes(self, H, W):
        return self.denoiser.calculate_indices_bytes(H, W)

    def configure(self, image, guidance_scale):
        H, W = image.shape[-2:]
        cond, uncond = self.conditioner(image)
        bitstream = self.denoiser.compress(cond)
        self.cond = self.denoiser.decompress(bitstream, H, W, image.device, uncond=uncond)
        self.guidance_scale = guidance_scale
        if guidance_scale is None:
            self.cond = self.uncond
        return bitstream

    def configure_decompress(self, bitstream, guidance_scale, H, W, device, uncond=None):
        self.cond = self.denoiser.decompress(bitstream, H, W, device, uncond=uncond)
        self.guidance_scale = guidance_scale
        if guidance_scale is None:
            self.cond = self.uncond

    def predict_noise(self, noisy_latent, timestep):
        """
        Predict noise in the latent at given timestep.
        
        Args:
            noisy_latent (torch.Tensor): Noisy latent tensor
            timestep (int): Current timestep
            
        Returns:
            torch.Tensor: Predicted noise in DDPM space
        """
        # Get current SNR for scaling
        snr = self.get_timestep_snr(timestep)

        # Get scaling factor to convert between DDPM and OT flow spaces
        ot_flow_to_ddpm_factor = get_ot_flow_to_ddpm_factor(snr)

        # Convert DDPM space latent to OT flow space
        ot_flow_latent = noisy_latent / ot_flow_to_ddpm_factor

        # calculate timestep for flow matching (revert)
        t_cur = 1 - torch.tensor([timestep / 1000], device=self.device, dtype=self.dtype)

        # Get prediction from denoiser (in OT flow space)
        with torch.no_grad():
            # prepare cfg input
            if self.guidance_scale is not None and self.guidance_scale > 1:
                cfg_x = torch.cat([ot_flow_latent, ot_flow_latent], dim=0)
                cfg_t = t_cur.repeat(2)
                cfg_cond = self.cond
            else:
                cfg_x = ot_flow_latent
                cfg_t = t_cur
                cfg_cond = self.cond[-1:]   # extract cond part

            out = self.denoiser(x=cfg_x.to(self.dtype), t=cfg_t.to(self.dtype), y=None, cond=cfg_cond.to(self.dtype))[0].to(torch.float32)

            # process cfg output
            if self.guidance_scale is not None and self.guidance_scale > 1:
                uncond_out, cond_out = out.chunk(2, dim=0)
                out = uncond_out + self.guidance_scale * (cond_out - uncond_out)

            # revert t direction
            ot_flow_noise_pred = - out 

        # TODO: this code is needlessly complicated, because I wanted to avoid doing math.
        # clean it up.
        # TODO: calculate x0 hat in OT flow space:
        sigma = 1 / (snr + 1)
        alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(snr)
        x0_hat = ot_flow_latent - sigma * ot_flow_noise_pred

        # back-calculate the DDPM noise pred from noisy_latent and x0_hat
        ddpm_noise_pred = (noisy_latent - alpha_prod_t**0.5 * x0_hat) / beta_prod_t**0.5
        # Convert prediction back to DDPM space
        #ddpm_noise_pred = ot_flow_noise_pred * ot_flow_to_ddpm_factor * (alpha_prod_t ** 0.5)

        return ddpm_noise_pred.to(noisy_latent.dtype)
