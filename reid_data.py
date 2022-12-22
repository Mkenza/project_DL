import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional
import torch
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms

class ReIdDataset(VisionDataset):
    """A PyTorch vision dataset compatible avec les modèles vision
    """

    def __init__(self, image_folder, transform=None, ids=None):
        """
        Args:
            root: image directory.
            image_folder: folder contenant les images des individus.
            transform: transformer for image.
        """

        self.image_folder = image_folder
        self.ids = [ name[:4] for name in os.listdir(image_folder)]
        self.transform = transform

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        image, id = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, index, id

    def get_raw_item(self, index):
        id = self.ids[index]
        path = os.listdir(self.image_folder)[index]
        image = Image.open(os.path.join(self.image_folder, path)).convert('RGB')

        return image, id

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Build mini-batch tensors from a list of (image, index, id) tuples.
    Args:
        data: list of (image, index, id) tuple.
            - image: torch tensor of shape (3, 128, 64).
    Returns:
        images: torch tensor of shape (batch_size, 3, 128, 64).
        ids: list of ids of the images of the batch
    """
    # Sort a data list by caption length
    
    images, indexes, ids = zip(*data)
    
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    return images, indexes, ids

def get_loader(image_folder, transform=transforms.ToTensor(),
                      batch_size=100, shuffle=True,
                      num_workers=0, ids=None, collate_fn=collate_fn):
    # Data loader
    dataset = ReIdDataset(image_folder=image_folder,
                                transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader