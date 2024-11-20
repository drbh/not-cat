import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from tqdm.auto import tqdm
import numpy as np


class NotCatDataset(Dataset):
    def __init__(
        self, cat_folder, non_cat_folder, max_samples=5000, preload=True, transform=None
    ):
        if transform:
            self.transform = transform
        else:
            # no op transform
            self.transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ]
            )

        # only load half the samples for cats (when balanced the models tend to overfit on cats)
        max_samples_cat = max_samples // 2
        cat_files = sorted(
            [os.path.join(cat_folder, f) for f in os.listdir(cat_folder)]
        )[:max_samples_cat]
        non_cat_files = sorted(
            [os.path.join(non_cat_folder, f) for f in os.listdir(non_cat_folder)]
        )[:max_samples]

        self.image_paths = [(p, 1) for p in cat_files] + [(p, 0) for p in non_cat_files]
        np.random.shuffle(self.image_paths)

        # preload images into memory for faster training
        self.preloaded = None
        if preload:
            print("Preloading images into memory...")
            self.preloaded = []
            for path, label in tqdm(self.image_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    img_tensor = self.transform(img)
                    self.preloaded.append(
                        (img_tensor, torch.tensor(label, dtype=torch.float32))
                    )
                except Exception:
                    continue
            print(f"Preloaded {len(self.preloaded)} images")

    def __len__(self):
        return (
            len(self.preloaded) if self.preloaded is not None else len(self.image_paths)
        )

    def __getitem__(self, idx):
        if self.preloaded is not None:
            return self.preloaded[idx]

        path, label = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return self.transform(image), torch.tensor(label, dtype=torch.float32)
        except Exception:
            return torch.zeros((3, 128, 128)), torch.tensor(label, dtype=torch.float32)
