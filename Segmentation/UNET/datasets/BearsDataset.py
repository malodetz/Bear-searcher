import torch
from torch.utils.data import Dataset

import glob
import os
import numpy as np
import cv2

from UNET.datasets.utils import get_color_map


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


def tensor_from_mask_image(mask: np.ndarray) -> torch.Tensor:
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, -1)
    return tensor_from_rgb_image(mask)


class BearsDataset(Dataset):
    def __init__(self, image_folder, transform, mask_folder=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

        self.images = BearsDataset.parse_folder(self.image_folder)
        self.color_map = get_color_map()

    @staticmethod
    def parse_folder(path):
        if path is None:
            return []
        images = glob.glob1(path,  '*.PNG')
        images.sort()

        return images

    @staticmethod
    def load_image(path) -> np.array:
        return cv2.imread(path, 0)

    @staticmethod
    def load_mask(path) -> np.array:
        return cv2.imread(path, 0)

    @staticmethod
    def split_grayscale_mask_into_channels_by_color_map(mask, color_map) -> torch.Tensor:
        masks = []

        for i in color_map.values():
            masks.append(mask == i)

        return torch.cat(masks).float()

    def mask_to_grayscale(self, masks) -> np.ndarray:
        masks = masks.cpu().numpy()

        colors_by_index = list(self.color_map.values())
        img = np.zeros(masks.shape[1:], dtype=np.uint8)

        for i in range(len(masks)):
            img[masks[i] == 1] = colors_by_index[i]

        return img

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_folder, image_name)

        image = BearsDataset.load_image(image_path)

        if self.mask_folder is None:
            sample = self.transform(image=image) ##
            image = sample['image'] ##
            return image_name, tensor_from_mask_image(image)

        mask_path = os.path.join(self.mask_folder, image_name)
        mask = BearsDataset.load_mask(mask_path)

        sample = self.transform(image=image, mask=mask) ##
        image, mask = sample['image'], sample['mask'] ##

        image = tensor_from_mask_image(image)
        image = torch.cat([image, image, image])
        mask = tensor_from_mask_image(mask)

        mask = BearsDataset.split_grayscale_mask_into_channels_by_color_map(mask, self.color_map)

        return image, mask

    def __len__(self):
        return len(self.images)
