"""
Generate translated MRI samples from a trained MMIT-DDPM model.

For each test slice, `sampling.num_ensemble` independent samples are produced and
saved to the output directory as PyTorch tensors (.pt files).

Configuration is managed by Hydra (configs/default.yaml). Override any value
from the command line, e.g.:

    python scripts/sampling_translation.py sampling.model_path=./results/savedmodel.pt \
        sampling.num_ensemble=5 sampling.use_ddim=true
"""
import os
import random
import sys
import time

sys.path.append(".")

import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf

from models import dist_util, logger
from models.bratsloader import load_data
from models.script_util import create_model_and_diffusion
from models.viz_util import get_viz

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 10
th.manual_seed(SEED)
th.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def visualize(img):
    _min = img.min()
    _max = img.max()
    return (img - _min) / (_max - _min + 1e-8)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    os.makedirs(cfg.sampling.output_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **OmegaConf.to_container(cfg.model, resolve=True),
        **OmegaConf.to_container(cfg.diffusion, resolve=True),
    )

    ds = load_data(
        data_dir=cfg.sampling.data_dir,
        batch_size=cfg.sampling.batch_size,
        image_size=cfg.model.image_size,
        test_flag=True,
        class_cond=cfg.model.class_cond,
    )
    data_loader = th.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    data = iter(data_loader)

    logger.log(f"loading model weights from {cfg.sampling.model_path}...")
    model.load_state_dict(
        dist_util.load_state_dict(cfg.sampling.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if cfg.model.use_fp16:
        model.convert_to_fp16()
    model.eval()

    all_images = []
    while len(all_images) * cfg.sampling.batch_size < cfg.sampling.num_samples:
        batch, _class, path = next(data)
        c = th.randn_like(batch[:, :1, ...])
        img = th.cat((batch, c), dim=1)

        slice_id = os.path.basename(path[0])

        get_viz().image(visualize(img[0, 0, ...]), opts=dict(caption="input_channel_0"))
        get_viz().image(visualize(img[0, 1, ...]), opts=dict(caption="input_channel_1"))

        logger.log("sampling...")

        model_kwargs = {}
        if cfg.model.class_cond:
            model_kwargs["y"] = _class["y"].to(dist_util.dev())

        sample_fn = (
            diffusion.p_sample_loop_known
            if not cfg.sampling.use_ddim
            else diffusion.ddim_sample_loop_known
        )

        for i in range(cfg.sampling.num_ensemble):
            t0 = time.perf_counter()
            sample, x_noisy, org = sample_fn(
                model,
                (cfg.sampling.batch_size, 2, cfg.model.image_size, cfg.model.image_size),
                img,
                clip_denoised=cfg.sampling.clip_denoised,
                model_kwargs=model_kwargs,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.log(f"ensemble {i}: {elapsed_ms:.1f} ms")

            out_path = os.path.join(cfg.sampling.output_dir, f"{slice_id}_{i}_output.pt")
            th.save(sample.clone().detach(), out_path)

        all_images.append(slice_id)


if __name__ == "__main__":
    main()
