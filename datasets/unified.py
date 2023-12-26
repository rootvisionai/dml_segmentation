"""
This code is from https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
"""

import cv2
import numpy as np
import torch
from torchvision.datasets import VOCSegmentation

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import sys, os

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

CLASSES = [
    'background',
    'live_knot',
    'death_know',
    'knot_missing',
    'knot_with_crack',
    'crack',
    'quartzity',
    'resin',
    'marrow',
    'blue_stain',
    'overgrown',
]

COLORMAP_OLD = [
    (0, 0, 0),
    (0, 255, 0),
    (255, 0, 0),
    (255, 100, 0),
    (255, 175, 0),
    (255, 0, 100),
    (100, 0, 100),
    (255, 0, 255),
    (0, 0, 255),
    (16, 255, 255),
    (0, 64, 0),
]

COLORMAP_NEW = [
    (0, 0, 0),
    (255, 0, 121),
    (101, 0, 255),
    (253, 255, 0),
    (0, 49, 255),
    (0, 186, 255),
    (0, 255, 39),
    (215, 0, 255),
    (0, 24, 255),
    (0, 157, 255),
    (0, 131, 255),
]

class DATASET(object):
    def __init__(
            self,
            transform=None,
            root="./datasets/wood_defect",
            image_set="train",
            task="diffusion_segmentation" # autoencoder, diffusion_segmentation
    ):
        self.transform = transform
        self.root = root
        self.image_set = image_set
        self.task = task
        self.get_records()

    def get_records(self):
        images_dir = os.path.join(self.root, "images")
        masks_dir = os.path.join(self.root, "masks")
        self.images = [os.path.join(images_dir, elm) for elm in os.listdir(images_dir)]
        self.masks = [os.path.join(masks_dir, elm) for elm in os.listdir(masks_dir)]
        self.images = self.images[0:int(0.8 * len(self.images))] if self.image_set == "train" else self.images[0:int(0.2 * len(self.images))]
        self.masks = self.masks[0:int(0.8 * len(self.masks))] if self.image_set == "train" else self.masks[0:int(0.2 * len(self.masks))]

    @staticmethod
    def _convert_to_new_labels(mask):
        for i, rgb_old in enumerate(COLORMAP_OLD):
            mask[np.all(mask == rgb_old, axis=-1)] = COLORMAP_NEW[i]
        return mask

    def __len__(self):
        return len(self.masks)

    @staticmethod
    def _convert_to_binary_mask(mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(COLORMAP_OLD)), dtype=np.float32)
        for label_index, label in enumerate(COLORMAP_OLD):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask

    @staticmethod
    def _convert_to_index_mask(mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, 1), dtype=np.int)
        for label_index, label in enumerate(COLORMAP_OLD):
            segmentation_mask[:, :, 0] = label_index*np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask

    def segmentation(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = self._convert_to_binary_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"] / 255
            mask = transformed["mask"]
            mask = mask.permute(2, 0, 1)
        return image, mask.to(torch.float32)

    def __getitem__(self, i):
        return self.segmentation(i)

def get_dataloader(
        root="./datasets/wood_defect",
        set_type="train",
        transform=None,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        task="diffusion_segmentation"
):

    set = DATASET(
        transform=transform,
        root=root,
        image_set=set_type,
        task=task
    )

    data_loader = DataLoader(
        set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return data_loader


def get_transforms(cfg, eval=True):
    train_transform = A.Compose(
        [
            A.Resize(cfg.input_size, cfg.input_size),

            A.ShiftScaleRotate(
                shift_limit=cfg.augmentations.ShiftScaleRotate.shift_limit,
                scale_limit=cfg.augmentations.ShiftScaleRotate.scale_limit,
                rotate_limit=cfg.augmentations.ShiftScaleRotate.rotate_limit,
                p=cfg.augmentations.ShiftScaleRotate.probability
            ),

            A.RGBShift(
                r_shift_limit=cfg.augmentations.RGBShift.r_shift_limit,
                g_shift_limit=cfg.augmentations.RGBShift.g_shift_limit,
                b_shift_limit=cfg.augmentations.RGBShift.b_shift_limit,
                p=cfg.augmentations.RGBShift.probability
            ),

            A.RandomBrightnessContrast(
                brightness_limit=cfg.augmentations.RandomBrightnessContrast.brightness_limit,
                contrast_limit=cfg.augmentations.RandomBrightnessContrast.contrast_limit,
                p=cfg.augmentations.RandomBrightnessContrast.probability
            ),
            ToTensorV2()
        ]
    )

    if eval:
        val_transform = A.Compose(
            [
                A.Resize(cfg.input_size, cfg.input_size),
                ToTensorV2()
            ]
        )

        return train_transform, val_transform

    else:
        return train_transform