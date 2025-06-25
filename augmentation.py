import torch
import torchvision.transforms as T
import random
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

# -----------------------------
# 1. Load dataset
# -----------------------------
data = torch.load("ModelStealingPub.pt", weights_only=False)
imgs = data.imgs  # Assuming this is a list of tensors or PIL images

# -----------------------------
# 2. Define individual augmentations
# -----------------------------
single_transforms = [
    T.RandomHorizontalFlip(p=1.0),
    T.RandomVerticalFlip(p=1.0),
    T.RandomRotation(degrees=15, expand=False),  # Keep size, may crop edges
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # Removed RandomResizedCrop because it changes size
]

# -----------------------------
# 3. Define full pipeline: ToPIL → One Random Aug → ToTensor
# -----------------------------
def apply_one_random_aug(img_tensor):
    if isinstance(img_tensor, torch.Tensor):
        img = T.ToPILImage()(img_tensor)
    else:
        img = img_tensor  # already PIL

    aug = random.choice(single_transforms)
    aug_img = aug(img)

    # Return PIL image directly
    return aug_img

# -----------------------------
# 4. Apply augmentation to each image
# -----------------------------
augmented_imgs = [apply_one_random_aug(img) for img in imgs]

# -----------------------------
# 5. Store only augmented images in a new .pt file
# -----------------------------

augmented_dataset = TaskDataset(transform=None)
augmented_dataset.imgs = augmented_imgs

torch.save(augmented_dataset, "ModelStealingPub_augmented.pt")
