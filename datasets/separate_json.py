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
import json

from itertools import permutations

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

color_numbers = list(range(32, 255, 32))
COLORMAP = list(permutations(color_numbers, 3))
print(len(COLORMAP))

class DATASET(object):
    def __init__(
            self,
            transform=None,
            root="./datasets/dataset_name",
            set_type=None
    ):
        global COLORMAP

        self.transform = transform
        self.root = root
        self.path_to_sets = os.path.join(root, set_type)
        self.get_records()

        self.label_map = None
        self.get_label_and_color_map()

        COLORMAP = COLORMAP[0:100]

    def get_label_and_color_map(self):
        with open(os.path.join(self.root, "label_map.json"), "r") as fp:
            self.label_map = json.load(fp)

    def get_records(self):
        images_dir = os.path.join(self.path_to_sets)
        annotations_dir = os.path.join(self.path_to_sets)
        self.annotations = [os.path.join(annotations_dir, elm) for elm in os.listdir(annotations_dir) if ".json" in elm]
        self.images = [os.path.join(images_dir, elm) for elm in os.listdir(images_dir) if any([k in elm for k in [".jpg", ".jpeg", ".JPG", ".png", ".PNG"]])]

    @staticmethod
    def _convert_to_new_labels(mask):
        for i, rgb_val in enumerate(COLORMAP):
            mask[np.all(mask == rgb_val, axis=-1)] = COLORMAP[i]
        return mask

    def __len__(self):
        return len(self.annotations)

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

    def load_mask(self, fullpath):
        with open(fullpath, "r") as fp:
            annotations = json.load(fp)
            points_labels = [(elm["points"], elm["label"]) for elm in annotations["shapes"]]
            image_height = annotations["imageHeight"]
            image_width = annotations["imageWidth"]
        
        mask = self.convert_points_and_labels_to_mask(points_labels, image_height, image_width)
        return mask
    
    def convert_points_and_labels_to_mask(self, points_labels, image_height, image_width):
        canvas_binary = np.zeros((image_height, image_width, max(list(self.label_map.values()))+1), dtype=np.uint8)
        
        for pt_lb in points_labels:
            canvas_temp = np.zeros((image_height, image_width, 1), dtype=np.uint8)
            
            pt = pt_lb[0]
            pt = np.array(pt).reshape(-1, 2).astype(np.int32)
            
            lb = pt_lb[1]
            lb_int = int(lb) # self.label_map[lb]
            try:
                color = COLORMAP[lb_int]
            except:
                print(lb, len(COLORMAP))
            cv2.fillPoly(canvas_temp, [pt], (255, 255, 255))
            row_coord, col_coord = np.where( (canvas_temp/255).mean(axis=2) == 1 )
            canvas_binary[row_coord, col_coord, lb_int] = 1
            
        return canvas_binary
    
    def segmentation(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.load_mask(self.annotations[index])
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
        root=root,
        set_type=set_type
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
            A.augmentations.crops.transforms.Crop(
                x_min=cfg.crop_image.x_min,
                y_min=cfg.crop_image.y_min, 
                x_max=cfg.crop_image.x_max,
                y_max=cfg.crop_image.y_max,
                p=cfg.crop_image.status
            ),
            
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
                A.augmentations.crops.transforms.Crop(
                    x_min=cfg.crop_image.x_min,
                    y_min=cfg.crop_image.y_min,
                    x_max=cfg.crop_image.x_max,
                    y_max=cfg.crop_image.y_max,
                    p=cfg.crop_image.status
                ),
                A.Resize(cfg.input_size, cfg.input_size),
                ToTensorV2()
            ]
        )

        return train_transform, val_transform

    else:
        return train_transform