import numpy as np
import torch
from torch.utils.data import Dataset

from src.augmentation import *
from src.utils import *


class ImageDataset(Dataset):
    def __init__(self, train_df, transforms=None):

        self.image_paths = train_df["image_path"].values
        self.labels = train_df["label"].values
        self.augmentations = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target = self.labels[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(target)
