"""
This code is from https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
"""

import cv2
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import os
import json

from itertools import permutations
from functools import wraps
import time

color_numbers = list(range(32, 255, 32))
COLORMAP = list(permutations(color_numbers, 3))

def timeit(func, debug=False):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {args[0]}.{func.__name__} Took {total_time:.4f} seconds')
        return result

    @wraps(func)
    def identity(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    if debug:
        return timeit_wrapper
    else:
        return identity

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
        self.set_type = set_type
        self.path_to_sets = os.path.join(root, set_type)

        self.label_map = None
        self.get_label_and_color_map()

        self.get_records()

        COLORMAP = COLORMAP[0:100]

    def get_label_and_color_map(self):
        with open(os.path.join(self.root, "label_map.json"), "r") as fp:
            self.label_map = json.load(fp)

    def get_records(self):
        images_dir = os.path.join(self.path_to_sets)
        annotations_dir = os.path.join(self.path_to_sets)
        self.annotations = [os.path.join(annotations_dir, elm) for elm in os.listdir(annotations_dir) if ".json" in elm]
        # self.annotations = self.select_records(first_n_labels=10)
        self.images = [os.path.join(images_dir, elm) for elm in os.listdir(images_dir) if any([k in elm for k in [".jpg", ".jpeg", ".JPG", ".png", ".PNG"]])]

    def select_records(self, first_n_labels=10):
        selected_labels = {key: self.label_map[key] for i, key in enumerate(self.label_map) if i < first_n_labels}
        annotations = []
        for annot_path in self.annotations:
            with open(annot_path, "r") as fp:
                annot = json.load(fp)
                if any([elm["label"] in selected_labels for elm in annot["shapes"]]):
                    annotations.append(annot_path)
        return annotations

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

    @timeit
    def load_mask(self, fullpath):
        with open(fullpath, "r") as fp:
            annotations = json.load(fp)
            points_labels = [(elm["points"], elm["label"]) for elm in annotations["shapes"]]
            image_height = annotations["imageHeight"]
            image_width = annotations["imageWidth"]
        
        mask = self.convert_points_and_labels_to_mask(points_labels, image_height, image_width)
        return mask

    @timeit
    def convert_points_and_labels_to_mask(self, points_labels, image_height, image_width):
        canvas_binary = np.zeros((image_height, image_width, max(list(self.label_map.values()))+1), dtype=np.uint8)
        
        for pt_lb in points_labels:
            canvas_temp = np.zeros((image_height, image_width, 1), dtype=np.uint8)
            
            pt = pt_lb[0]
            pt = np.array(pt).reshape(-1, 2).astype(np.int32)
            
            lb = pt_lb[1]
            lb_int = self.label_map[lb]
            cv2.fillPoly(canvas_temp, [pt], (255, 255, 255))
            row_coord, col_coord = np.where( (canvas_temp/255).mean(axis=2) == 1 )
            canvas_binary[row_coord, col_coord, lb_int] = 1
            
        return canvas_binary

    @timeit
    def segmentation(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = self.load_mask(self.annotations[index])

        transformed = self.transform[0](image=img, mask=msk)
        image = transformed["image"] / 255
        mask = transformed["mask"]
        mask = mask.permute(2, 0, 1)

        # if self.set_type == "train":
        #     transformed_ = self.transform[1](image=img, mask=msk)
        #     image_ = transformed_["image"] / 255
        #     mask_ = transformed_["mask"]
        #     mask_ = mask_.permute(2, 0, 1)
        #     return image, mask, image_, mask_
        # else:
        return image, mask

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
            PadToSquare(),
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
                PadToSquare(),
                A.Resize(cfg.input_size, cfg.input_size),
                ToTensorV2()
            ]
        )

        return train_transform, val_transform

    else:
        return train_transform

class PadToSquare(object):
    @timeit
    def __call__(self, **kwargs):
        for key in kwargs:
            shape = kwargs[key].shape
            row, column = shape[0], shape[1]

            if row > column:
                temp = np.zeros((row, row, kwargs[key].shape[-1]), dtype=np.uint8)
                start = int((row - column)/2)
                temp[:, start:start+column, :] = kwargs[key]
            else:
                temp = np.zeros((column, column, kwargs[key].shape[-1]), dtype=np.uint8)
                start = int((column - row)/2)
                temp[start:start+row, :, :] = kwargs[key]

            kwargs[key] = temp
        return kwargs