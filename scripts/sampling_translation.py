"""
Generate translated MRI samples from a trained MMIT-DDPM model.

For each test slice, `num_ensemble` independent samples are produced and saved
to the output directory as PyTorch tensors (.pt files).
"""
import argparse
import os
import random
import sys
import time

sys.path.append(".")

import numpy as np
import torch as th

from models import dist_util, logger
from models.bratsloader import load_data
from models.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
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


def main():
    args = create_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        test_flag=True,
        class_cond=args.class_cond,
    )
    data_loader = th.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    data = iter(data_loader)

    logger.log(f"loading model weights from {args.model_path}...")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        batch, _class, path = next(data)
        c = th.randn_like(batch[:, :1, ...])
        img = th.cat((batch, c), dim=1)

        # Cross-platform: use os.path.basename instead of splitting on backslash.
        slice_id = os.path.basename(path[0])

        get_viz().image(visualize(img[0, 0, ...]), opts=dict(caption="input_channel_0"))
        get_viz().image(visualize(img[0, 1, ...]), opts=dict(caption="input_channel_1"))

        logger.log("sampling...")

        model_kwargs = {}
        if args.class_cond:
            model_kwargs["y"] = _class["y"].to(dist_util.dev())

        sample_fn = (
            diffusion.p_sample_loop_known
            if not args.use_ddim
            else diffusion.ddim_sample_loop_known
        )

        for i in range(args.num_ensemble):
            t0 = time.perf_counter()
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.log(f"ensemble {i}: {elapsed_ms:.1f} ms")

            out_path = os.path.join(args.output_dir, f"{slice_id}_{i}_output.pt")
            th.save(sample.clone().detach(), out_path)

        all_images.append(slice_id)


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        output_dir="./results",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./results/savedmodel.pt",
        num_ensemble=5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()