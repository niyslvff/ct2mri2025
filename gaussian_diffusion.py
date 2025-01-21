import torch
import torch.nn.functional as F
import numpy as np


class DDPMScheduler():
    def __init__(self, num_train_timesteps=1000, start_step=0.0001, end_step=0.02):
        self.T = num_train_timesteps
        self.betas = torch.linspace(start=start_step, end=end_step, steps=num_train_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the base_diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.Tensor(np.append(np.array(self.posterior_variance[1]), np.array(self.posterior_variance[1:])))
        )

        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev / (1.0 - self.alphas_cumprod))
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )

    @staticmethod
    def get_index_from_list(arr, timesteps, broadcast_shape):
        batch_size = timesteps.shape[0]
        out = arr.gather(-1, timesteps.cpu())
        return out.reshape(batch_size, *((1,) * (len(broadcast_shape) - 1))).to(timesteps.device)


    def forward_diffusion_sample(self, x_0, t, device='cpu'):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(
            device) * noise.to(device), noise.to(device)

    def resample(self, noise_pred, x, t):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def denoise_ddim(self, noise_pred, x, t, t_prev):
        alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod, t, x.shape)
        alphas_cumprod_t_prev = self.get_index_from_list(self.alphas_cumprod, t_prev, x.shape)

        x0_pred = alphas_cumprod_t_prev.sqrt() / alphas_cumprod_t.sqrt() * (
                x - (1 - alphas_cumprod_t).sqrt() * noise_pred)
        if t_prev == 0:
            return x0_pred
        else:
            noise = torch.randn_like(x)
            dir_xt = (1 - alphas_cumprod_t_prev).sqrt() * noise
            return x0_pred + dir_xt

    @torch.no_grad()
    def resample_to_origin_from_t(self, noise_pred, model, img, t):
        img = self.resample(noise_pred, img, t)
        img = torch.clamp(img, -1.0, 1.0)
        for i in range(0, t)[::-1]:
            t = torch.full((1,), i, device=img.device, dtype=torch.long)
            noise_pred = model(img, t)
            img = self.resample(noise_pred, img, t)
            img = torch.clamp(img, -1.0, 1.0)
        return img
