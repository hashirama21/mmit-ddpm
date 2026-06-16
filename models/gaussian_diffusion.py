"""
Gaussian diffusion process for MMIT-DDPM.

Ported from Ho et al. (https://github.com/hojonathanho/diffusion) and adapted
for conditional multi-modal medical image translation.
"""
import enum
import math

import numpy as np
import torch as th
import torch

from .viz_util import get_viz
from .train_util import visualize
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def standardize(img):
    mean = th.mean(img)
    std = th.std(img)
    return (img - mean) / std


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Pre-defined beta schedules.

    Schedules remain numerically stable in the limit of num_diffusion_timesteps
    and should not be changed once committed (backwards-compatibility).
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Discretize a given alpha_t_bar function into a beta schedule.

    :param alpha_bar: callable t ∈ [0,1] → cumulative product of (1-beta).
    :param max_beta: upper bound to prevent singularities near 1.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()   # model predicts x_{t-1}
    START_X = enum.auto()      # model predicts x_0
    EPSILON = enum.auto()      # model predicts ε


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self in (LossType.KL, LossType.RESCALED_KL)


# ---------------------------------------------------------------------------
# GaussianDiffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param betas: 1-D numpy array of betas for each diffusion timestep.
    :param model_mean_type: what the model predicts (ε, x_0, or x_{t-1}).
    :param model_var_type: how variance is parameterized.
    :param loss_type: which loss function to use.
    :param rescale_timesteps: if True, pass timesteps scaled to [0, 1000].
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_mean_variance(self, x_start, t):
        """q(x_t | x_0): mean, variance and log_variance."""
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0) by adding t steps of Gaussian noise."""
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """q(x_{t-1} | x_t, x_0): posterior mean and variance."""
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t) and a prediction of x_0.

        Loss and sampling operate only on the last channel (the translation
        target); the preceding channels are the conditioning source modality.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x.shape[0]
        # Only the last channel is the diffusion target; C is fixed to 1.
        C = 1
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # Work only on the target channel from here on.
        x = x[:, -1:, ...]

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    # ------------------------------------------------------------------
    # Internal prediction utilities
    # ------------------------------------------------------------------

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            ) * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def condition_mean(self, cond_fn, p_mean_var, x, t, org, model_kwargs=None):
        """Classifier guidance via Sohl-Dickstein et al. (2015)."""
        a, gradient = cond_fn(x, self._scale_timesteps(t), org, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return a, new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """Classifier guidance via Song et al. (2020)."""
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps.detach()

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x.detach(), t.detach(), eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out, eps

    # ------------------------------------------------------------------
    # DDPM sampling
    # ------------------------------------------------------------------

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """Sample x_{t-1} from the model at timestep t."""
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x[:, -1:, ...])
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self, model, shape, noise=None, clip_denoised=True,
        denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False,
    ):
        """Generate samples from the model (DDPM)."""
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, cond_fn=cond_fn,
            model_kwargs=model_kwargs, device=device, progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_known(
        self, model, shape, img, org=None, noise=None, clip_denoised=True,
        denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False,
    ):
        """Sample with a known source image; noise only the target channel."""
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        img = img.to(device)
        target_noise = th.randn_like(img[:, :1, ...])
        # Concatenate source channels with a fresh noise channel as the target.
        x_noisy = torch.cat((img[:, :-1, ...], target_noise), dim=1)

        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, noise=x_noisy,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn,
            cond_fn=cond_fn, org=org, model_kwargs=model_kwargs,
            device=device, progress=progress,
        ):
            final = sample

        return final["sample"], x_noisy, img

    def p_sample_loop_progressive(
        self, model, shape, time=None, noise=None, clip_denoised=True,
        denoised_fn=None, cond_fn=None, org=None, model_kwargs=None,
        device=None, progress=False,
    ):
        """
        Yield intermediate samples from each DDPM denoising step.

        The source channels are re-concatenated at every step so that the
        model always receives the conditioning context.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        img = noise if noise is not None else th.randn(*shape, device=device)
        T = self.num_timesteps if time is None else int(time)
        indices = list(range(T))[::-1]

        # Preserve source channels across denoising steps.
        n_source = img.shape[1] - 1
        org_MRI = img[:, :n_source, ...]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            if i % 100 == 0:
                get_viz().image(
                    visualize(img.cpu()[0, -1, ...]),
                    opts=dict(caption=f"sample_step_{i}"),
                )

            # Re-prepend source channels if only the target channel remains.
            if img.shape[1] == 1:
                img = torch.cat((org_MRI, img), dim=1)

            with th.no_grad():
                out = self.p_sample(
                    model, img.float(), t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
            yield out
            img = out["sample"]

    # ------------------------------------------------------------------
    # DDIM sampling
    # ------------------------------------------------------------------

    def ddim_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None,
        cond_fn=None, model_kwargs=None, eta=0.0,
    ):
        """Sample x_{t-1} using DDIM."""
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        if cond_fn is not None:
            out, _ = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = th.randn_like(x[:, -1:, ...])
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0,
    ):
        """Encode x_{t} to x_{t+1} using the DDIM reverse ODE."""
        assert eta == 0.0, "Reverse ODE requires a deterministic path (eta=0)"
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self, model, shape, noise=None, clip_denoised=True, denoised_fn=None,
        cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0,
        start_timestep=None,
    ):
        """
        Generate samples using DDIM.

        :param start_timestep: t to begin denoising from. Defaults to num_timesteps - 1
                               (full denoising from pure noise).
        """
        if device is None:
            device = next(model.parameters()).device
        b = shape[0]
        T = start_timestep if start_timestep is not None else self.num_timesteps - 1
        t = th.full((b,), T, device=device, dtype=th.long)

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model, shape, time=t, noise=noise,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn,
            cond_fn=cond_fn, model_kwargs=model_kwargs,
            device=device, progress=progress, eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_known(
        self, model, shape, img, clip_denoised=True, denoised_fn=None,
        cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0,
        start_timestep=None,
    ):
        """DDIM sampling with a known source image."""
        if device is None:
            device = next(model.parameters()).device
        b = shape[0]
        T = start_timestep if start_timestep is not None else self.num_timesteps - 1
        t = th.full((b,), T, device=device, dtype=th.long)

        img = img.to(device)
        target_noise = th.randn_like(img[:, :1, ...])
        x_noisy = torch.cat((img[:, :-1, ...], target_noise), dim=1).float()

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model, shape, time=t, noise=x_noisy,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn,
            cond_fn=cond_fn, model_kwargs=model_kwargs,
            device=device, progress=progress, eta=eta,
        ):
            final = sample

        return final["sample"], x_noisy, img

    def ddim_sample_loop_progressive(
        self, model, shape, time=1000, noise=None, clip_denoised=True,
        denoised_fn=None, cond_fn=None, model_kwargs=None,
        device=None, progress=False, eta=0.0,
    ):
        """Yield intermediate samples from each DDIM denoising step."""
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        img = noise if noise is not None else th.randn(*shape, device=device)

        if isinstance(time, th.Tensor):
            T = int(time[0].item())
        else:
            T = int(time)
        indices = list(range(T))[::-1]

        n_source = img.shape[1] - 1
        org_source = img[:, :n_source, ...]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            if img.shape[1] == 1:
                img = torch.cat((org_source, img), dim=1).float()

            with th.no_grad():
                out = self.ddim_sample(
                    model, img, t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
            yield out
            img = out["sample"]

    def ddim_sample_loop_interpolation(
        self, model, shape, img1, img2, lambdaint, noise=None, clip_denoised=True,
        denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False,
    ):
        """Latent-space interpolation between two images via DDIM."""
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(499, 500, (b,), device=device).long()

        img1 = torch.tensor(img1).to(device)
        img2 = torch.tensor(img2).to(device)
        shared_noise = th.randn_like(img1).to(device)

        x_noisy1 = self.q_sample(x_start=img1, t=t, noise=shared_noise)
        x_noisy2 = self.q_sample(x_start=img2, t=t, noise=shared_noise)
        interp = lambdaint * x_noisy1 + (1 - lambdaint) * x_noisy2

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model, shape, time=t, noise=interp,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn,
            cond_fn=cond_fn, model_kwargs=model_kwargs,
            device=device, progress=progress,
        ):
            final = sample
        return final["sample"], interp, img1, img2

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------

    def training_losses_translation(
        self, model, classifier, x_start, t, model_kwargs=None, noise=None
    ):
        """
        Compute training losses for one timestep batch.

        Noise is only added to the last channel (translation target); the
        preceding channels are the fixed source conditioning.

        :return: (terms_dict, model_output)
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start[:, -1:, ...])

        target = x_start[:, -1:, ...]
        target_noisy = self.q_sample(target, t, noise=noise)

        # Clone to avoid mutating the original batch tensor.
        x_t = x_start.clone().float()
        x_t[:, -1:, ...] = target_noisy.float()

        terms = {}

        if self.loss_type in (LossType.MSE, LossType.RESCALED_MSE):
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in (ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE):
                B = x_t.shape[0]
                C = 1  # single target channel
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=target,
                    x_t=target_noisy,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target_label = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=target, x_t=target_noisy, t=t
                )[0],
                ModelMeanType.START_X: target,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            terms["mse"] = mean_flat((target_label - model_output) ** 2)
            terms["loss"] = terms["mse"] + terms.get("vb", 0)
        else:
            raise NotImplementedError(self.loss_type)

        return terms, model_output

    # ------------------------------------------------------------------
    # Variational lower-bound
    # ------------------------------------------------------------------

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        """One term of the variational lower-bound in bits-per-dim."""
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def _prior_bpd(self, x_start):
        """Prior KL term of the VLB, in bits-per-dim."""
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """Compute the full variational lower-bound over all timesteps."""
        device = x_start.device
        batch_size = x_start.shape[0]

        vb, xstart_mse, mse = [], [], []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            with th.no_grad():
                out = self._vb_terms_bpd(
                    model, x_start=x_start, x_t=x_t, t=t_batch,
                    clip_denoised=clip_denoised, model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices and broadcast.

    :param arr: 1-D numpy array.
    :param timesteps: tensor of indices into arr.
    :param broadcast_shape: shape [batch, 1, …] to broadcast to.
    :return: tensor of shape broadcast_shape.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
