"""
Train the MMIT-DDPM model for multi-modal MRI translation.

Configuration is managed by Hydra (configs/default.yaml). Override any value
from the command line, e.g.:

    python scripts/train_translation.py training.lr=2e-5 model.num_channels=64
"""
import sys

sys.path.append(".")

import hydra
import torch as th
from omegaconf import DictConfig, OmegaConf

from models import dist_util, logger
from models.bratsloader import load_data
from models.resample import create_named_schedule_sampler
from models.script_util import create_model_and_diffusion
from models.train_util import TrainLoop


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **OmegaConf.to_container(cfg.model, resolve=True),
        **OmegaConf.to_container(cfg.diffusion, resolve=True),
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        cfg.training.schedule_sampler, diffusion, maxt=diffusion.num_timesteps
    )

    logger.log("creating data loader...")
    dataset = load_data(
        data_dir=cfg.training.data_dir,
        batch_size=cfg.training.batch_size,
        image_size=cfg.model.image_size,
        class_cond=cfg.model.class_cond,
    )
    data_loader = th.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=th.cuda.is_available(),
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        dataloader=data_loader,
        batch_size=cfg.training.batch_size,
        microbatch=cfg.training.microbatch,
        lr=cfg.training.lr,
        ema_rate=cfg.training.ema_rate,
        log_interval=cfg.training.log_interval,
        save_interval=cfg.training.save_interval,
        resume_checkpoint=cfg.training.resume_checkpoint,
        use_fp16=cfg.training.use_fp16,
        fp16_scale_growth=cfg.training.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.training.weight_decay,
        lr_anneal_steps=cfg.training.lr_anneal_steps,
    ).run_loop()


if __name__ == "__main__":
    main()
