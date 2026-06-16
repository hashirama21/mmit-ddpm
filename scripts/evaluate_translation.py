"""
Evaluate a trained MMIT-DDPM model on paired slices.

Unlike sampling, evaluation needs the ground-truth target, so it reads paired
files shaped (H, W, 2) (channel 0 = source, channel 1 = target), translates the
source, and reports PSNR / SSIM / MAE against the real target channel.

    python scripts/evaluate_translation.py eval.model_path=./results/savedmodel.pt \
        eval.data_dir=./data/training diffusion.timestep_respacing=50
"""
import csv
import os
import sys

sys.path.append(".")

import hydra
import nibabel as nib
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models import dist_util, logger
from models.bratsloader import _list_nifti_files_recursively
from models.script_util import create_model_and_diffusion


def _to_unit(x):
    """Map a tensor from [-1, 1] to [0, 1] for metric computation."""
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def _load_pair(path):
    """Load a paired slice and return (source, target) as 224x224 float arrays in [-1, 1]."""
    image = nib.load(path).get_fdata()
    if np.count_nonzero(image) == 0:
        norm = image
    else:
        norm = 2.0 * (image - image.min()) / (image.max() - image.min()) - 1.0
    norm = norm[8:-8, 8:-8, :]
    return norm[..., 0].astype(np.float32), norm[..., 1].astype(np.float32)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    dist_util.setup_dist()
    logger.configure()

    model, diffusion = create_model_and_diffusion(
        **OmegaConf.to_container(cfg.model, resolve=True),
        **OmegaConf.to_container(cfg.diffusion, resolve=True),
    )
    model.load_state_dict(
        dist_util.load_state_dict(cfg.eval.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    sample_fn = (
        diffusion.ddim_sample_loop_known
        if cfg.eval.use_ddim
        else diffusion.p_sample_loop_known
    )

    files = _list_nifti_files_recursively(cfg.eval.data_dir)
    if cfg.eval.num_eval:
        files = files[: cfg.eval.num_eval]
    logger.log(f"evaluating {len(files)} slice(s) from {cfg.eval.data_dir}...")

    rows, psnrs, ssims, maes = [], [], [], []
    for path in files:
        source, target = _load_pair(path)
        src = th.tensor(source, device=dist_util.dev())[None, None]
        img = th.cat((src, th.randn_like(src)), dim=1)

        sample, _, _ = sample_fn(
            model,
            (1, 2, src.shape[-2], src.shape[-1]),
            img,
            clip_denoised=cfg.eval.clip_denoised,
            model_kwargs={},
        )
        pred = _to_unit(sample[0, -1].cpu().numpy())
        gt = _to_unit(target)

        psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim = structural_similarity(gt, pred, data_range=1.0)
        mae = float(np.mean(np.abs(gt - pred)))
        psnrs.append(psnr)
        ssims.append(ssim)
        maes.append(mae)
        rows.append((os.path.basename(path), psnr, ssim, mae))
        logger.log(f"{os.path.basename(path)}: PSNR={psnr:.2f} SSIM={ssim:.4f} MAE={mae:.4f}")

    os.makedirs(os.path.dirname(cfg.eval.output_csv) or ".", exist_ok=True)
    with open(cfg.eval.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "psnr", "ssim", "mae"])
        writer.writerows(rows)

    logger.log(
        f"=== mean over {len(files)} slices === "
        f"PSNR={np.mean(psnrs):.2f}  SSIM={np.mean(ssims):.4f}  MAE={np.mean(maes):.4f}"
    )
    logger.log(f"per-slice metrics written to {cfg.eval.output_csv}")


if __name__ == "__main__":
    main()
