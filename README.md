# MMIT-DDPM – Multilateral Medical Image Translation with Class and Structure Supervised Diffusion-Based Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/mmit-ddpm/blob/main/model_training.ipynb)

Official PyTorch implementation of the paper [MMIT-DDPM – Multilateral medical image translation with class and structure supervised diffusion-based model](https://doi.org/10.1016/j.compbiomed.2024.109501), published in *Computers in Biology and Medicine*.

## Data

MMIT-DDPM was evaluated on the BraTS dataset. The data directory must follow this structure:

```
data
├── training
│   ├── brats_train_XXXXX_ft1ce_XX_w.nii.gz
│   ├── brats_train_XXXXX_ft2_XX_w.nii.gz
│   ├── brats_train_XXXXX_t1cet2_XX_w.nii.gz
│   ├── brats_train_XXXXX_t1cetf_XX_w.nii.gz
│   ├── brats_train_XXXXX_t2f_XX_w.nii.gz
│   ├── brats_train_XXXXX_t2t1ce_XX_w.nii.gz
│   └── ...
└── testing
    ├── brats_test_XXXXX_ft1ce_XX_w.nii.gz
    └── ...
```

Each slice is `240×240`. Training files have shape `(H, W, 2)` — channel 0 = source modality, channel 1 = target modality. Testing files hold the source modality only, with shape `(H, W)`. The loader re-normalises every slice to `[-1, 1]` and centre-crops it to `224×224`. `XXXXX` and `XX` denote the patient ID and slice index respectively.

MMIT-DDPM supports six translation pairs across three MRI modalities (Flair, T1CE, T2): `ft1ce` · `ft2` · `t1cet2` · `t1cetf` · `t2f` · `t2t1ce`

The full dataset is available on Synapse: [syn51514132](https://www.synapse.org/Synapse:syn51514132).

## Configuration

All settings live in [`configs/default.yaml`](configs/default.yaml) and are managed with [Hydra](https://hydra.cc). Edit the file, or override any value on the command line using `group.key=value` syntax.

## Training

```bash
git clone https://github.com/hashirama21/mmit-ddpm.git
cd mmit-ddpm
pip install -r requirements.txt
```

Launch training with the defaults:

```bash
python scripts/train_translation.py
```

Override values as needed, e.g.:

```bash
python scripts/train_translation.py training.batch_size=10 training.lr=1e-4 model.num_channels=128
```

Checkpoints are saved in the `results/` folder.

## Sampling

Generate an ensemble of translated outputs:

```bash
python scripts/sampling_translation.py \
  sampling.model_path=./results/savedmodel.pt \
  sampling.num_ensemble=5
```

Outputs are saved in the `results/` folder.

## Evaluation

Evaluation needs the ground-truth target, so it runs on paired slices (`(H, W, 2)`). It translates the source channel and reports PSNR / SSIM / MAE against the real target, writing per-slice metrics to a CSV:

```bash
python scripts/evaluate_translation.py \
  eval.model_path=./results/savedmodel.pt \
  eval.data_dir=./data/training \
  diffusion.timestep_respacing=100
```

## Notebook

[`model_training.ipynb`](model_training.ipynb) runs the full pipeline end-to-end (download → clean → visualise → smoke-test → train → sample → evaluate → save weights), and opens directly in Colab via the badge above.