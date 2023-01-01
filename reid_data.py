import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional
from pandas.core.frame import is_dataclass
import torch
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from itertools import chain

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
        # get anchor
        image, id = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)
        ids = [id]

        # we add 2 positive examples only to training batches
        if "test" in self.image_folder:
          images_p = image.unsqueeze(dim=0)

          indexes = [index]

        else:
          images = []

          # get indexes of positive examples
          positive = np.where(np.array(self.ids)==id)

          # choose randomly 2 other examples
          indexes_pos = np.random.choice(positive[0], size=2)
          for idx in indexes_pos:
                # get positive image
                image_pos, _ = self.get_raw_item(idx)

                # transform it if required
                if self.transform is not None:
                    image_pos = self.transform(image_pos)
                # add it to the list of images
                images.append(image_pos)
                ids.append(id)
        
          # keep the same order for indexes and images
          images.append(image)
          indexes = np.append(indexes_pos, index)

          # stack them to get a tensor of (3, 3, 148, 64)
          images_p = torch.stack(images, 0)
        return images_p, indexes, ids

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
        data: list of (images, indexes, ids) tuple.
            - images: torch tensor of shape (3, 3, 128, 64).
    Returns:
        images: torch tensor of shape (batch_size*3, 3, 128, 64).
        ids: list of ids of the images of the batch
    """
    # Sort a data list by caption length
    
    images, indexes, ids = zip(*data)
    
    # Merge mini-batches of images 
    images = torch.stack(images, 0) #dim = (batch_size, 3, 3, 128, 64)
    images_p = images.flatten(start_dim=0, end_dim=1) #dim = (batch_size * 3, 3, 128, 64)

    # flatten tuple of lists to one tuple of ids (resp. indexes) of the batch
    ids_flatten = tuple(chain.from_iterable(ids))
    indexes_flattened = tuple(chain.from_iterable(indexes))
    return images_p, indexes_flattened , ids_flatten

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