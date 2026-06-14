# MMIT-DDPM – Multilateral Medical Image Translation with Class and Structure Supervised Diffusion-Based Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/mmit-ddpm/blob/main/05-synt1ce-pix2pix.ipynb)

uOfficial PyTorch implementation of the paper [MMIT-DDPM – Multilateral medical image translation with class and structure supervised diffusion-based model](https://doi.org/10.1016/j.compbiomed.2024.109501), published in *Computers in Biology and Medicine*.

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

Each `.nii.gz` file contains a 2D slice of shape `(H, W, 2)` — channel 0 = source modality, channel 1 = target modality, pre-normalised to [-1, 1]. `XXXXX` and `XX` denote the patient ID and slice index respectively.

MMIT-DDPM supports six translation pairs across three MRI modalities (Flair, T1CE, T2): `ft1ce` · `ft2` · `t1cet2` · `t1cetf` · `t2f` · `t2t1ce`

The full dataset is available at: [MMIT-DDPM Data](https://drive.google.com/file/d/1dLTVF7-oBhpBqriIkJ9GPYizrzcwKhJR/view?usp=sharing)

## Training

```bash
git clone https://github.com/hashirama21/mmit-ddpm.git
cd mmit-ddpm
pip install -r requirements.txt
```

Set the model and diffusion flags:

```bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
```

Launch training:

```bash
python scripts/train_translation.py --data_dir ./data/training $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```

Checkpoints are saved in the `results/` folder.

## Sampling

Generate an ensemble of 5 translated outputs:

```bash
python scripts/sampling_translation.py \
  --data_dir ./data/testing \
  --model_path ./results/savedmodel.pt \
  --num_ensemble 5 \
  $MODEL_FLAGS $DIFFUSION_FLAGS
```

Outputs are saved in the `results/` folder.