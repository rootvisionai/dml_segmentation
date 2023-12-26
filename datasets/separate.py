"""
This code is from https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
"""

import cv2
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import sys, os

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

CLASSES = [
    'background',
    'ventilation'
]

COLORMAP = [
    (0, 0, 0),
    (255, 255, 255)
]

class DATASET(object):
    def __init__(
            self,
            transform=None,
            root="./datasets/dataset_name",
    ):
        self.transform = transform
        self.root = root
        self.get_records()

    def get_records(self):
        images_dir = os.path.join(self.root, "images")
        masks_dir = os.path.join(self.root, "masks")
        self.masks = [os.path.join(masks_dir, elm) for elm in os.listdir(masks_dir) if ".png" in elm]
        self.images = [os.path.join(images_dir, elm) for elm in os.listdir(images_dir) if ".jpg" in elm]
#         self.images = [os.path.join(images_dir, elm) for elm in os.listdir(images_dir) if os.path.join(images_dir, elm).replace("masks", "images") in self.masks]

    @staticmethod
    def _convert_to_new_labels(mask):
        for i, rgb_old in enumerate(COLORMAP):
            mask[np.all(mask == rgb_old, axis=-1)] = COLORMAP[i]
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
        segmentation_mask = np.zeros((height, width, len(COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(COLORMAP):
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
        for label_index, label in enumerate(COLORMAP):
            segmentation_mask[:, :, 0] = label_index*np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask

    def segmentation(self, index):
#         print("--->", self.images[index])
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         print(self.images[index], "---", self.images[index].replace("images", "masks"))
        mask = cv2.imread(self.masks[index])
        mask = self._convert_to_binary_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"] / 255
            mask = transformed["mask"]
            mask = mask.permute(2, 0, 1)
        return image, mask.to(torch.long)

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
):

    set = DATASET(
        transform=transform,
        root=os.path.join(root, set_type),
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
            A.augmentations.crops.transforms.Crop(x_min=450, y_min=0, x_max=1930, y_max=2048, always_apply=True),
            
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
                A.augmentations.crops.transforms.Crop(x_min=450, y_min=0, x_max=1930, y_max=2048, always_apply=True),
                A.Resize(cfg.input_size, cfg.input_size),
                ToTensorV2()
            ]
        )

        return train_transform, val_transform

    else:
        return train_transform