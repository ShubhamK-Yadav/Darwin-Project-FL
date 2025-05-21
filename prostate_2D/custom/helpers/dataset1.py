# custom/helpers/dataset.py
import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class PICAIProstateDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, case_ids=None, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.case_ids = case_ids if case_ids else sorted([
            f.replace(".nii.gz", "") for f in os.listdir(image_dir) if f.endswith(".nii.gz")
        ])
        self.transforms = transforms

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        img = nib.load(os.path.join(self.image_dir, f"{case_id}.nii.gz")).get_fdata()
        img = img.astype(np.float32)

        if self.label_dir:
            lbl = nib.load(os.path.join(self.label_dir, f"{case_id}.nii.gz")).get_fdata().astype(np.int64)
        else:
            lbl = np.zeros(img.shape[1:], dtype=np.int64)  # one label per slice

        if self.transforms:
            # Add MONAI transforms later
            sample = self.transforms({"image": img, "label": lbl})
            img, lbl = sample["image"], sample["label"]

        return torch.tensor(img), torch.tensor(lbl)

