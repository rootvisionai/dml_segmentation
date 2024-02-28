import random

from PIL import Image
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
random.seed(7)
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

                padding = {"y": start, "x": 0}
            else:
                temp = np.zeros((column, column, kwargs[key].shape[-1]), dtype=np.uint8)
                start = int((column - row)/2)
                temp[start:start+row, :, :] = kwargs[key]

                padding = {"y": 0, "x": start}

            kwargs[key] = temp

        kwargs["padding"] = padding
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

        last_epoch = self.model.load_checkpoint(checkpoint_path, device=self.device, load_opt=False)

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

    def load_supports(self, image_paths, mask_paths):

        images = []
        masks = []
        for image_path, mask_path in zip(image_paths, mask_paths):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = self.load_mask(mask_path)
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"] / 255
            mask = transformed["mask"]
            mask = mask.permute(2, 0, 1)

            images.append(image)
            masks.append(mask)

        self.support_image = torch.stack(images, dim=0)
        self.support_mask = torch.stack(masks, dim=0)

    def load_query(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_size = image.shape[0:2][::-1]

        transformed = self.transform(image=image)
        self.padding = transformed["padding"]

        self.original_size = (max(self.original_size), max(self.original_size))

        image = transformed["image"] / 255
        return image.unsqueeze(0)

    def process(self, query_image_path):
        query_image = self.load_query(query_image_path)
        out, probs = self.model.predict_few_shot(
            self.support_image,
            self.support_mask,
            query_image,
            k=self.cfg.inference.num_of_neighbors,
            device="cuda",
            return_probs=True
        )
        return out, probs

    def postprocess(self, inference_output, probs):
        inference_output = inference_output.squeeze(1)
        colored_image = COLORMAP.to(self.device)[inference_output.long()][0].to(torch.uint8).cpu().numpy()
        probs = (probs * 255).squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()

        try:
            colored_image = cv2.resize(colored_image, self.original_size)
            colored_image = colored_image[
                            self.padding["x"]:colored_image.shape[0]-self.padding["x"],
                            self.padding["y"]:colored_image.shape[1]-self.padding["y"],
                            :]

            probs = cv2.resize(probs, self.original_size)
            probs = probs[
                   self.padding["x"]:probs.shape[0] - self.padding["x"],
                   self.padding["y"]:probs.shape[1] - self.padding["y"]
                   ]

        except cv2.error as e:
            print("Error resizing image:", e)
            print("Image shape:", colored_image.shape)
            print("Target size:", self.original_size)
            raise

        return colored_image, probs


if __name__ == "__main__":
    import time
    import glob
    import os
    import json

    checkpoint_dir = "arch[FPN]-backbone[timm-regnetx_032]-pretrained_weights[imagenet]-out_layer_size[512]-in_channels[3]-version[2]"
    support_images = ["./inference_data/support/VIN-WBA21BX0107P11721_U06_21BX_Image-0001_d05350d6-b1ef-4aaf-b7de-733b3192a7f2.jpg"]

    support_exts = ["." + support_image.split(".")[-1] for support_image in support_images]
    support_annotations = [support_image.replace(support_exts[i], ".json") for i, support_image in enumerate(support_images)]
    category = "P07-front"
    query_images = []
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    for ext in exts:
        query_images += glob.glob(os.path.join("inference_data", "query", category, "*"+ext))

    cfg = load_config(os.path.join(
        "logs",
        checkpoint_dir,
        "config.yml"
    ))

    checkpoint_path = os.path.join(
        "logs",
        f"{checkpoint_dir}",
        "ckpt_finetuned.pth"
    )

    instance = Inference(cfg, checkpoint_path=checkpoint_path, label_map_path=os.path.join("inference_data", "label_map.json"))
    instance.load_supports(image_paths=support_images, mask_paths=support_annotations)

    for k, query_image in enumerate(query_images):
        st = time.time()
        output, probs = instance.process(query_image_path=query_image)
        colored, probs = instance.postprocess(output, probs)

        os.makedirs("outputs", exist_ok=True)
        qimg = cv2.imread(query_image)
        qimg = cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB)
        size = qimg.shape[0:2][::-1]
        colored = cv2.resize(colored, size)
        colored = colored/255
        qimg = qimg / 255

        colored = ((colored * 0.9) + (qimg * 0.5))/2
        colored = colored * 255
        colored = np.asarray(colored, dtype=np.uint8)

        inferenced_image_name = query_image.split("\\")[-1].replace(".", "_inferenced.")
        Image.fromarray(colored).save(fp=os.path.join("outputs", f"{inferenced_image_name}"))
        # cv2.imwrite(filename=os.path.join("outputs", f"{query_image}"), img=colored)
        # cv2.imwrite(filename=os.path.join("outputs", "probs.jpg"), img=probs)
        et = time.time()
        print(f"Processed {query_image} | {et - st}")