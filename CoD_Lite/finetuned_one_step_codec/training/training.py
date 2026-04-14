import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

from cod.utils.no_grad import no_grad
from cod.utils.model_loader import load_pretrained_state_dict


# ============================================================
#  BaseModelLoss
# ============================================================

class BaseModelLoss(nn.Module):
    """Dummy model loss that returns zero."""
    def forward(self, y, net):
        net_recon = net.train_step(y)
        out = dict(net_loss=torch.zeros_like(y).mean())
        return net_recon, out


# ============================================================
#  OneStepCoDModelLoss
# ============================================================

class OneStepCoDModelLoss(BaseModelLoss):

    def __init__(
            self,
            feat_loss_weight: float = 0.5,
            encoder: nn.Module = None,
            align_layer=8,
            proj_denoiser_dim=256,
            proj_hidden_dim=256,
            proj_encoder_dim=256,
            net_loss_ckpt_path=None,
            pretrain_prefix="diffusion_trainer.",
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.feat_loss_weight = feat_loss_weight
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
        self.pretrain_prefix = pretrain_prefix
        if net_loss_ckpt_path is not None:
            self.load_pretrained(net_loss_ckpt_path)

    def load_pretrained(self, net_loss_ckpt_path, pretrained_ema=True):
        assert net_loss_ckpt_path is not None
        print(f"Loading Network Loss weights from {net_loss_ckpt_path} (ema=True, strict=True)")
        pretrained_dict = load_pretrained_state_dict(net_loss_ckpt_path)
        proj_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith(f"{self.pretrain_prefix}proj."):
                proj_dict[k[len(f"{self.pretrain_prefix}proj."):]] = v
        self.proj.load_state_dict(proj_dict, strict=True)

    def forward(self, y, net):
        src_feature = []
        handle = net.blocks[self.align_layer - 1].register_forward_hook(
            lambda net, input, output: src_feature.append(output))
        net_recon, codec_res = net.train_step(y)

        src_feature = src_feature[0]
        if len(src_feature.shape) == 4:
            src_feature = src_feature.flatten(2, 3).transpose(1, 2)
        src_feature = self.proj(src_feature)
        handle.remove()

        with torch.no_grad():
            dst_feature = self.encoder(y)
        if dst_feature.shape[1] != src_feature.shape[1]:
            src_feature = src_feature[:, :dst_feature.shape[1]]

        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_feature, dim=-1)
        cos_loss = 1 - cos_sim

        out = dict(
            cos=cos_loss.mean(),
            vq=codec_res["vq_loss"].mean(),
            net_loss=self.feat_loss_weight * cos_loss.mean() + 0.25 * codec_res["vq_loss"].mean(),
        )
        return net_recon, out


# ============================================================
#  CoDPerceptualLossTrainer
# ============================================================

from cod.diffusion.diffusion import time_shift_fn


class CoDPerceptualLossTrainer(nn.Module):
    def __init__(
            self,
            lognorm_t=False,
            timeshift=1.0,
            net_loss: nn.Module = None,
            lpips_vgg_weight=1.0,
            lpips_alex_weight=0.5,
            dmd_cfg=3.0,
            dmd_loss_weight=2.0,
            gan_loss_weight=0.01,
            net_loss_weight=1.0,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.timeshift = timeshift
        self.net_loss = net_loss
        self.lpips_vgg_weight = lpips_vgg_weight
        self.lpips_alex_weight = lpips_alex_weight
        self.dmd_cfg = dmd_cfg
        self.dmd_loss_weight = dmd_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.net_loss_weight = net_loss_weight

        self.lpips_fn_vgg = lpips.LPIPS(net='vgg').eval()
        self.lpips_fn_alex = lpips.LPIPS(net='alex').eval()
        no_grad(self.lpips_fn_vgg)
        no_grad(self.lpips_fn_alex)

        self.pre_calculated_uncond = None

    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def __call__(self, net, real_net, fake_net, disc_net, x, condition, uncondition, metadata=None, dmd_step=False):
        if self.pre_calculated_uncond is None:
            self.pre_calculated_uncond = real_net.y_embedder(uncondition[:1])[0]

        if dmd_step:
            for n, p in fake_net.named_parameters():
                p.requires_grad = True
            for n, p in disc_net.named_parameters():
                p.requires_grad = True
            for n, p in net.named_parameters():
                p.requires_grad = False
            return self.guidance_step(net, real_net, fake_net, disc_net, x, condition, uncondition, metadata)
        else:
            for n, p in fake_net.named_parameters():
                p.requires_grad = False
            for n, p in disc_net.named_parameters():
                p.requires_grad = False
            for n, p in net.named_parameters():
                p.requires_grad = True
            return self.generator_step(net, real_net, fake_net, disc_net, x, condition, uncondition, metadata)

    def get_dmd_timestep(self, batch_size, device, dtype):
        if self.lognorm_t:
            base_t = torch.randn((batch_size), device=device, dtype=torch.float32).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=device, dtype=torch.float32)
        dmd_t = time_shift_fn(base_t, self.timeshift).to(dtype).view(-1, 1, 1, 1)
        return dmd_t

    def guidance_step(self, net, real_net, fake_net, disc_net, x, condition, uncondition, metadata=None):
        batch_size, c, height, width = x.shape

        with torch.no_grad():
            net_recon = net.inference(condition)

        dmd_noise = torch.randn_like(x)
        dmd_t = self.get_dmd_timestep(batch_size, x.device, x.dtype)

        dmd_x_t = dmd_t * net_recon + (1 - dmd_t) * dmd_noise
        dmd_v_t = net_recon - dmd_noise

        with torch.no_grad():
            fake_cond = real_net.y_embedder(condition)[0]
        fake_out, _ = fake_net(dmd_x_t, dmd_t, y=None, cond=fake_cond)

        fm_loss = (fake_out - dmd_v_t) ** 2

        d_fake_logits = disc_net(net_recon)
        d_real_logits = disc_net(x)
        d_fake_loss = F.relu(1. + d_fake_logits)
        d_real_loss = F.relu(1. - d_real_logits)

        out = dict(
            fake_fm=fm_loss.mean(),
            d_fake=d_fake_loss.mean(),
            d_real=d_real_loss.mean(),
            fake_loss=fm_loss.mean() + 0.5 * (d_fake_loss.mean() + d_real_loss.mean()),
        )
        return out

    def generator_step(self, net, real_net, fake_net, disc_net, x, condition, uncondition, metadata=None):
        batch_size, c, height, width = x.shape

        net_recon, net_loss_dict = self.net_loss(condition, net)

        l1_loss = (net_recon - x).abs()
        lpips_loss = self.lpips_vgg_weight * self.lpips_fn_vgg(net_recon, x) + \
                     self.lpips_alex_weight * self.lpips_fn_alex(net_recon, x)

        dmd_noise = torch.randn_like(x)
        dmd_t = self.get_dmd_timestep(batch_size, x.device, x.dtype)
        dmd_x_t = dmd_t * net_recon + (1 - dmd_t) * dmd_noise

        dmd_cfg = self.dmd_cfg

        with torch.no_grad():
            cond = fake_net.y_embedder(condition)[0]
            uncond = self.pre_calculated_uncond.repeat(batch_size, 1, 1, 1)

            if dmd_cfg > 1.0:
                dmd_x_t = torch.cat([dmd_x_t, dmd_x_t], dim=0)
                dmd_t = torch.cat([dmd_t, dmd_t], dim=0)
                dmd_cond = torch.cat([cond, uncond], dim=0)
            else:
                dmd_cond = cond
            real_out, _ = real_net(dmd_x_t, dmd_t, y=None, cond=dmd_cond)

            if dmd_cfg > 1.0:
                real_out_cond, real_out_uncond = real_out.chunk(2, dim=0)
                real_out = real_out_uncond + dmd_cfg * (real_out_cond - real_out_uncond)
                dmd_x_t = dmd_x_t[:batch_size]
                dmd_t = dmd_t[:batch_size]
                dmd_cond = dmd_cond[:batch_size]
            real_out_x = dmd_x_t + (1 - dmd_t) * real_out

            fake_out, _ = fake_net(dmd_x_t, dmd_t, y=None, cond=dmd_cond)
            fake_out_x = dmd_x_t + (1 - dmd_t) * fake_out

            p_real = (net_recon - real_out_x)
            p_fake = (net_recon - fake_out_x)

            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = torch.nan_to_num(grad)

        dmd_loss = (net_recon.float() - (net_recon - grad).detach().float()) ** 2

        g_fake_logits = disc_net(net_recon)
        g_fake_loss = -g_fake_logits

        net_loss = net_loss_dict['net_loss']
        del net_loss_dict['net_loss']

        out = net_loss_dict
        out.update(
            l1=l1_loss.mean(),
            lpips=lpips_loss.mean(),
            dmd=dmd_loss.mean(),
            g_fake=g_fake_loss.mean(),
            loss=l1_loss.mean() + lpips_loss.mean() +
                 self.dmd_loss_weight * dmd_loss.mean() +
                 self.gan_loss_weight * g_fake_loss.mean() +
                 self.net_loss_weight * net_loss.mean()
        )
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self.net_loss.state_dict(
            destination=destination,
            prefix=prefix + "net_loss.",
            keep_vars=keep_vars)
        return destination
