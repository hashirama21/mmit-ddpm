import os

import blobfile as bf
import nibabel as nib
import numpy as np
import torch
import torch.nn


def load_data(data_dir, batch_size, image_size, test_flag=False, class_cond=False):
    """
    Build a BRATSDataset from a directory tree of NIfTI files.

    File naming convention: brats_{split}_{id}_{modality_pair}_{slice}_w.nii.gz
    The modality token (field index 3) is used as the class label when class_cond=True.

    :param data_dir: root directory containing .nii / .nii.gz files.
    :param batch_size: unused here; kept for a consistent interface with DataLoader callers.
    :param image_size: unused here; cropping is fixed to 224×224.
    :param test_flag: if True, return (tensor, dict, path); if False return (tensor, dict).
    :param class_cond: if True, populate the "y" key with an integer class label.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = _list_nifti_files_recursively(data_dir)
    classes = None

    if class_cond:
        class_names = [os.path.basename(p).split("_")[3] for p in all_files]
        sorted_classes = {name: i for i, name in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[n] for n in class_names]

    return BRATSDataset(all_files, classes, test_flag)


def _list_nifti_files_recursively(data_dir):
    """Return all NIfTI files under data_dir, sorted for determinism."""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        if bf.isdir(full_path):
            results.extend(_list_nifti_files_recursively(full_path))
        elif full_path.endswith((".nii", ".nii.gz")):
            results.append(full_path)
    return results


class BRATSDataset(torch.utils.data.Dataset):
    """
    Dataset for paired BraTS NIfTI slices.

    Training files are 3-channel (H×W×C) volumes where C encodes the
    source/target modality pair.  The loader permutes them to (C, H, W).

    Test files are 2-D (H×W) slices returned as (1, H, W) tensors.
    """

    def __init__(self, all_files, classes, test_flag, shard=0, num_shards=1):
        super().__init__()
        self.local_images = all_files[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.test_flag = test_flag

    def __getitem__(self, idx):
        path = self.local_images[idx]
        image = nib.load(path).get_fdata()

        if np.count_nonzero(image) == 0:
            norm = image
        else:
            norm = 2.0 * (image - image.min()) / (image.max() - image.min()) - 1.0

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        if not self.test_flag:
            # Training: volume shape (H, W, C) → crop → permute to (C, H, W)
            norm = norm[..., 8:-8, 8:-8, :]
            return torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1), out_dict
        else:
            # Test: 2-D slice shape (H, W) → crop → unsqueeze to (1, H, W)
            norm = norm[..., 8:-8, 8:-8]
            return torch.tensor(norm, dtype=torch.float32).unsqueeze(0), out_dict, path

    def __len__(self):
        return len(self.local_images)
