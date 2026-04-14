import logging
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, List, Callable
from einops import rearrange
from cod.utils.no_grad import no_grad

logger = logging.getLogger(__name__)


# ============================================================
#  Guidance
# ============================================================

def simple_guidance_fn(out, cfg):
    uncondition, condtion = out.chunk(2, dim=0)
    out = uncondition + cfg * (condtion - uncondition)
    return out


# ============================================================
#  Scheduling
# ============================================================

class LinearScheduler:
    def alpha(self, t) -> Tensor:
        return (t).view(-1, 1, 1, 1)
    def sigma(self, t) -> Tensor:
        return (1-t).view(-1, 1, 1, 1)
    def dalpha(self, t) -> Tensor:
        return torch.full_like(t, 1.0).view(-1, 1, 1, 1)
    def dsigma(self, t) -> Tensor:
        return torch.full_like(t, -1.0).view(-1, 1, 1, 1)

    def dalpha_over_alpha(self, t) -> Tensor:
        return self.dalpha(t) / self.alpha(t)
    def dsigma_mul_sigma(self, t) -> Tensor:
        return self.dsigma(t)*self.sigma(t)
    def w(self, t):
        return self.sigma(t)


# ============================================================
#  Step Functions
# ============================================================

def shift_respace_fn(t, shift=3.0):
    return t / (t + (1 - t) * shift)

def ode_step_fn(x, v, dt, s, w):
    return x + v * dt

def time_shift_fn(t, timeshift=1.0):
    return t/(t+(1-t)*timeshift)


# ============================================================
#  Training: REPATrainer
# ============================================================

class REPATrainer(nn.Module):
    def __init__(
            self,
            feat_loss_weight: float=0.5,
            one_step_mse_weight: float=0.0,
            null_condition_p: float=0.1,
            lognorm_t=False,
            timeshift=1.0,
            encoder:nn.Module=None,
            align_layer=8,
            proj_denoiser_dim=256,
            proj_hidden_dim=256,
            proj_encoder_dim=256,
            recon_dim=[3, 16, 16],
            *args,
            **kwargs
    ):
        super().__init__()
        self.null_condition_p = null_condition_p
        self.lognorm_t = lognorm_t
        self.timeshift = timeshift
        self.feat_loss_weight = feat_loss_weight
        self.one_step_mse_weight = one_step_mse_weight
        self.align_layer = align_layer
        self.encoder = encoder
        no_grad(self.encoder)

        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )
        self.proj_codec = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )
        self.recon_codec = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(proj_denoiser_dim, proj_hidden_dim, 1),
                nn.SiLU(),
                nn.Conv2d(proj_hidden_dim, proj_hidden_dim, 1),
                nn.SiLU(),
                nn.Conv2d(proj_hidden_dim, recon_dim[0] * recon_dim[1] * recon_dim[2], 1),
            )
        )
        self.recon_dim = recon_dim

    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def __call__(self, net, ema_net, solver, x, condition, uncondition, metadata=None):
        raw_images = metadata["raw_image"]
        batch_size, c, height, width = x.shape

        # part 1 : stadard repa training
        mask = torch.rand((batch_size), device=condition.device) < self.null_condition_p
        mask = mask.view(-1, *([1] * (len(condition.shape) - 1))).to(condition.dtype)
        y = condition*(1-mask)  + uncondition*mask

        if self.lognorm_t:
            base_t = torch.randn((batch_size), device=x.device, dtype=torch.float32).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift).to(x.dtype).view(-1, 1, 1, 1)
        noise = torch.randn_like(x)

        x_t = t * x + (1 - t) * noise
        v_t = (x - x_t) / (1 - t).clamp_min(net.v_clamp_min)

        src_feature = []
        def forward_hook(net, input, output):
            feature = output
            if isinstance(feature, tuple):
                feature = feature[0] # mmdit
            src_feature.append(feature)
        if getattr(net, "encoder", None) is not None:
            handle = net.encoder.blocks[self.align_layer - 1].register_forward_hook(forward_hook)
        else:
            handle = net.blocks[self.align_layer - 1].register_forward_hook(forward_hook)

        out, codec_out, codec_res = net(x_t, t, y, cond=None, return_codec_res=True)

        src_feature = src_feature[0]
        if len(src_feature.shape) == 4:
            src_feature = src_feature.flatten(2, 3).transpose(1, 2)
        src_feature = self.proj(src_feature)

        handle.remove()

        with torch.no_grad():
            dst_feature = self.encoder(raw_images)

        if dst_feature.shape[1] != src_feature.shape[1]:
            src_feature = src_feature[:, :dst_feature.shape[1]]

        codec_cos_loss = torch.nn.functional.cosine_similarity(self.proj_codec(codec_out.flatten(2, 3).transpose(1, 2)), dst_feature, dim=-1)
        codec_cos_loss = 1 - codec_cos_loss

        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_feature, dim=-1)
        cos_loss = 1 - cos_sim

        codec_recon = self.recon_codec(codec_out)
        codec_recon = rearrange(
            codec_recon,
            'b (c ph pw) h w -> b c (h ph) (w pw)',
            c=self.recon_dim[0], ph=self.recon_dim[1], pw=self.recon_dim[2]
        )
        codec_recon_loss = (codec_recon - x) ** 2

        fm_loss = (out - v_t)**2

        # do not apply on unconditional
        codec_cos_loss = codec_cos_loss * (1 - mask)[:, None]
        codec_recon_loss = codec_recon_loss * (1 - mask)[:, None, None, None]

        # part 2 : one-step mse training
        if self.one_step_mse_weight > 0:
            t0 = torch.zeros((batch_size), device=x.device, dtype=x.dtype)
            v0, _ = net(noise, t0, y=None, cond=codec_out)
            x0 = noise + v0
            one_step_mse = (x0 - x) ** 2
        else:
            one_step_mse = torch.zeros_like(fm_loss)

        out = dict(
            fm_loss=fm_loss.mean(),
            codec_recon_loss=codec_recon_loss.mean(),
            codec_cos_loss=codec_cos_loss.mean(),
            cos_loss=cos_loss.mean(),
            one_step_mse=one_step_mse.mean(),
            vq_loss=codec_res["vq_loss"].mean(),
            loss=fm_loss.mean() + codec_recon_loss.mean() + self.feat_loss_weight*(cos_loss.mean() + codec_cos_loss.mean()) + 0.25 * codec_res["vq_loss"].mean() + self.one_step_mse_weight * one_step_mse.mean(),
        )
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}

        self.proj.state_dict(
            destination=destination,
            prefix=prefix + "proj.",
            keep_vars=keep_vars)

        self.proj_codec.state_dict(
            destination=destination,
            prefix=prefix + "proj_codec.",
            keep_vars=keep_vars)

        self.recon_codec.state_dict(
            destination=destination,
            prefix=prefix + "recon_codec.",
            keep_vars=keep_vars)

        return destination


# ============================================================
#  Sampling: EulerSampler
# ============================================================

class EulerSampler(nn.Module):
    def __init__(
            self,
            scheduler: LinearScheduler = None,
            guidance_fn: Callable = None,
            num_steps: int = 250,
            guidance: Union[float, List[float]] = 1.0,
            w_scheduler: LinearScheduler = None,
            timeshift=1.0,
            guidance_interval_min: float = 0.0,
            guidance_interval_max: float = 1.0,
            step_fn: Callable = ode_step_fn,
            last_step=None,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__()
        self.num_steps = num_steps
        self.guidance = guidance
        self.guidance_fn = guidance_fn
        self.scheduler = scheduler
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None:
            if self.step_fn == ode_step_fn:
                logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, net, noise, condition, uncondition, codec_cond=None):
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        x = noise
        if codec_cond is None:
            cfg_condition = torch.cat([uncondition, condition], dim=0)
        else:
            cfg_condition = None
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0

            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            out, codec_cond = net(cfg_x, cfg_t, cfg_condition, codec_cond)
            if t_cur[0] > self.guidance_interval_min and t_cur[0] < self.guidance_interval_max:
                guidance = self.guidance
                out = self.guidance_fn(out, guidance)
            else:
                out = self.guidance_fn(out, 1.0)
            v = out
            s = ((1/dalpha_over_alpha)*v - x)/(sigma**2 - (1/dalpha_over_alpha)*dsigma_mul_sigma)
            if i < self.num_steps -1 :
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)
        return x
