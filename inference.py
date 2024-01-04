import os
import json
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import models
from utils import load_config

from itertools import permutations
from random import shuffle

color_numbers = list(range(32, 255, 32))
COLORMAP = list(permutations(color_numbers, 3))
shuffle(COLORMAP)
COLORMAP = torch.tensor(COLORMAP)

class PadToSquare(object):
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

class Inference:
    def __init__(self, cfg, checkpoint_path, label_map_path):
        self.cfg = cfg
        self.device = cfg.inference.device
        # get model
        model = models.load(cfg)
        model = model.to(torch.float32)
        self.model = model.to(self.device)
        self.model.eval()

        last_epoch = self.model.load_checkpoint(checkpoint_path, device=cfg.training.device, load_opt=False)

        print(
            f"Loaded model: {checkpoint_path}\n",
            f"Last epoch: {last_epoch}"
        )

        self.transform = self.define_transform()
        self.get_labels(label_map_path)

    def define_transform(self):
        val_transform = A.Compose(
            [
                A.augmentations.crops.transforms.Crop(
                    x_min=self.cfg.data.crop_image.x_min,
                    y_min=self.cfg.data.crop_image.y_min,
                    x_max=self.cfg.data.crop_image.x_max,
                    y_max=self.cfg.data.crop_image.y_max,
                    p=self.cfg.data.crop_image.status
                ),
                PadToSquare(),
                A.Resize(self.cfg.data.input_size, self.cfg.data.input_size),
                ToTensorV2()
            ]
        )
        return val_transform

    def get_labels(self, path):
        with open(path, "r") as fp:
            self.label_map = json.load(fp)

    def convert_points_and_labels_to_mask(self, points_labels, image_height, image_width):
        canvas_binary = np.zeros((image_height, image_width, max(list(self.label_map.values())) + 1), dtype=np.uint8)

        for pt_lb in points_labels:
            canvas_temp = np.zeros((image_height, image_width, 1), dtype=np.uint8)

            pt = pt_lb[0]
            pt = np.array(pt).reshape(-1, 2).astype(np.int32)

            lb = pt_lb[1]
            lb_int = self.label_map[lb]
            cv2.fillPoly(canvas_temp, [pt], (255, 255, 255))
            row_coord, col_coord = np.where((canvas_temp / 255).mean(axis=2) == 1)
            canvas_binary[row_coord, col_coord, lb_int] = 1

        return canvas_binary

    def load_mask(self, fullpath):
        with open(fullpath, "r") as fp:
            annotations = json.load(fp)
            points_labels = [(elm["points"], elm["label"]) for elm in annotations["shapes"]]
            image_height = annotations["imageHeight"]
            image_width = annotations["imageWidth"]

        mask = self.convert_points_and_labels_to_mask(points_labels, image_height, image_width)
        return mask

    def load_inputs(self, image_path, annotation_path=None):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_size = image.shape[0:2][::-1]

        if annotation_path:
            mask = self.load_mask(annotation_path)
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"] / 255
            mask = transformed["mask"]
            mask = mask.permute(2, 0, 1)
            return image.unsqueeze(0), mask.unsqueeze(0)

        else:
            transformed = self.transform(image=image)
            image = transformed["image"] / 255
            return image.unsqueeze(0)

    def define_support(self, image_path, annot_path):
        self.support_image, self.support_mask = self.load_inputs(image_path=image_path, annotation_path=annot_path)

    def process(self, query_image_path):
        query_image = self.load_inputs(query_image_path, annotation_path=None)
        out = self.model.predict_few_shot(self.support_image, self.support_mask, query_image, k=self.cfg.inference.num_of_neighbors, device="cuda")
        return out

    def postprocess(self, inference_output):
        inference_output = inference_output.squeeze(1)
        colored_image = COLORMAP.to(self.device)[inference_output][0].to(torch.uint8).cpu().numpy()

        try:
            colored_image = cv2.resize(colored_image, self.original_size)
        except cv2.error as e:
            print("Error resizing image:", e)
            print("Image shape:", colored_image.shape)
            print("Target size:", self.original_size)
            raise

        return colored_image


if __name__ == "__main__":
    checkpoint_dir = "arch[FPN]-backbone[timm-regnetx_040]-pretrained_weights[imagenet]-out_layer_size[256]-in_channels[3]-version[1]"

    support_image = "./inference_data/support/emblem_rain.jpg"
    support_annotation = "./inference_data/support/emblem_rain.json"
    query_image = "./inference_data/query/emblem_1.jpg"

    cfg = load_config(os.path.join(
        "logs",
        checkpoint_dir,
        "config.yml"
    ))

    checkpoint_path = os.path.join(
        "logs",
        f"{checkpoint_dir}",
        "ckpt.pth"
    )

    instance = Inference(cfg, checkpoint_path=checkpoint_path, label_map_path=os.path.join("inference_data", "label_map.json"))
    instance.define_support(image_path=support_image, annot_path=support_annotation)
    output = instance.process(query_image_path=query_image)
    colored = instance.postprocess(output)
    cv2.imwrite(filename="colored_output.jpg", img=colored)

