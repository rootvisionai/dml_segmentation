import random

from PIL import Image
import cv2
import torch
import torchvision
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

# COLORMAP = [[0, 0, 0], [255, 0, 255]]
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
    def __init__(self, cfg, checkpoint_path):
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
        self.support_ready = False

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

    def convert_points_and_labels_to_mask(self, points_labels, image_height, image_width):
        self.label_map = {}
        labels = set([elm[1] for elm in points_labels])
        for i, lb in enumerate(labels, start=0):
            self.label_map[lb] = i

        canvas_binary = np.zeros((image_height, image_width, max(list(self.label_map.values()))+1), dtype=np.uint8)

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

    def unique_elements_single_row(self, t, threshold=0.9):

        t = t.cpu()
        # Find elements that are unique to a single row
        unq_elms = torch.unique(t)
        unqs_probs_multi_class = []
        for unq_elm in unq_elms:
            if unq_elm != 0:
                unq_matrix = torch.where(t == unq_elm, torch.tensor(1., device=t.device), torch.tensor(0., device=t.device))
                probs = unq_matrix.sum(dim=1)/unq_matrix.shape[1]
                unqs = torch.where(probs > threshold, torch.tensor(1., device=t.device), torch.tensor(0., device=t.device))
                unqs_probs = torch.stack([unqs, probs], dim=1)
                unqs_probs_multi_class.append(unqs_probs)

        # here will be single class for now, if you see this comment, the algorithm works for single class
        unqs_probs = unqs_probs_multi_class[0]
        # will change here later so that it supports multi-class
        return unqs_probs

    def define_support(self, si, sm, device, negative_ratio):
        s_emb = torch.nn.functional.normalize(
            self.model.flatten(
                self.model.forward_all(si.to(torch.float32).to(device))
            ),
            2, 1
        )

        sm = self.model.flatten(sm.to(torch.int8).to(device)).flatten()
        neg_indices = torch.where(sm == 0)
        pos_indices = torch.where(sm == 1)
        n_size = pos_indices[0].shape[0] * negative_ratio
        n_max = neg_indices[0].shape[0]
        n = torch.randint(low=0, high=n_max - 1, size=(n_size,))
        neg_indices = neg_indices[0][n]
        pos_indices = pos_indices[0]

        s_emb = torch.cat([s_emb[neg_indices], s_emb[pos_indices]], dim=0)
        sm = torch.cat([sm[neg_indices], sm[pos_indices]], dim=0)
        self.support_ready = True
        self.s_emb = s_emb
        self.sm = sm

    def predict_few_shot(self, si, sm, qi, k=1, threshold=0.9, device="cpu", negative_ratio=100):

        with torch.no_grad():
            bs, _, row, col = qi.shape

            if not self.support_ready:
                self.define_support(si, sm, device, negative_ratio)

            q_emb = torch.nn.functional.normalize(
                self.model.flatten(
                    self.model.forward_all(qi.to(torch.float32).to(device))
                ),
                2, 1
            )

            cos_sim = torch.nn.functional.linear(q_emb, self.s_emb)
            cos_sim = torch.where(cos_sim < threshold, 0., cos_sim)

            topk = cos_sim.topk(1 + k)
            Y = self.sm[topk[1][:, 1:]]
            Y = Y.view(-1, Y.shape[-1])

            unqs_cnts = self.unique_elements_single_row(Y, threshold=threshold)

            output = unqs_cnts[:, 0]
            probs = unqs_cnts[:, 1]

            probs = probs.reshape((bs, 1, row, col))
            output = output.reshape((bs, 1, row, col))

            torchvision.utils.save_image(probs, fp="./probs.png")

        return output, probs

    def process(self, query_image):
        out, _ = self.predict_few_shot(
            si=self.support_image,  # torch.nn.functional.interpolate(self.support_image, (int(self.support_image.shape[-2]/2), int(self.support_image.shape[-1]/2)), mode="nearest"),
            sm=self.support_mask,  # torch.nn.functional.interpolate(self.support_mask, (int(self.support_mask.shape[-2]/2), int(self.support_mask.shape[-2]/2)), mode="nearest"),
            qi=query_image,  # torch.nn.functional.interpolate(query_image, (int(query_image.shape[-2]/2), int(query_image.shape[-2]/2)), mode="nearest"),
            k=self.cfg.inference.num_of_neighbors,
            threshold=self.cfg.inference.threshold,
            device="cuda",
            negative_ratio=self.cfg.inference.negative_samples_ratio
        )
        return out

    def postprocess(self, inference_output):
        inference_output = inference_output
        inference_output = torch.nn.functional.interpolate(inference_output, size=self.original_size, mode="nearest")
        inference_output = inference_output[
                           :, :,
                           self.padding["x"]:inference_output.shape[2] - self.padding["x"],
                           self.padding["y"]:inference_output.shape[3] - self.padding["y"],
                        ]
        inference_output = inference_output.squeeze(0)
        return inference_output

def visualize_and_save(query_image_path, inference_output, folder_to_save="output", mask_weight=0.8):
    colored_output = COLORMAP.to("cpu")[inference_output.long().to("cpu")][0].to(torch.uint8).numpy()

    os.makedirs(folder_to_save, exist_ok=True)
    qimg = cv2.imread(query_image_path)

    qimg = np.asarray(cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB), float)
    size = qimg.shape[0:2][::-1]

    colored_output = cv2.resize(colored_output, size)
    colored_output = np.asarray(colored_output, float)
    colored_output = ((colored_output * mask_weight) + (qimg * (1 - mask_weight))) / 2
    colored_output = np.asarray(colored_output, dtype=np.uint8)

    query_image_name = query_image_path.split("\\")[-1] if "\\" in query_image_path else query_image_path.split("/")[-1]
    Image.fromarray(colored_output).save(fp=os.path.join(folder_to_save, f"{query_image_name}"))

    return os.path.join(folder_to_save, f"{query_image_name}")

if __name__ == "__main__":
    import time
    import glob
    import os
    import json

    checkpoint_dir = "arch[Unet]-backbone[resnet50]-pretrained_weights[imagenet]-out_layer_size[512]-in_channels[3]-version[coco_proxy_opt]"
    support_image = [
        "./inference_data/support/emblem_1.jpg",
        # "./inference_data/support/800px-2010_brown_BMW_530i_rear.jpg",
    ]
    support_annotation = [
        "./inference_data/support/emblem_1.json",
        # "./inference_data/support/800px-2010_brown_BMW_530i_rear.json",
    ]

    out_folder = "bmw_emblems_0"
    query_images = glob.glob(os.path.join("inference_data", "query", out_folder, "*.jpg"))
    query_images += glob.glob(os.path.join("inference_data", "query", out_folder, "*.jpeg"))
    query_images += glob.glob(os.path.join("inference_data", "query", out_folder, "*.png"))

    cfg = load_config(os.path.join(
        "logs",
        checkpoint_dir,
        "config.yml"
    ))

    checkpoint_path = os.path.join(
        "logs",
        checkpoint_dir,
        "ckpt_finetuned.pth"
    )

    instance = Inference(cfg, checkpoint_path=checkpoint_path)
    instance.load_supports(image_paths=support_image, mask_paths=support_annotation)

    for k, query_image_path in enumerate(query_images):
        print(f"Processing {query_image_path}")
        st = time.time()

        query_image = instance.load_query(image_path=query_image_path)
        output = instance.process(query_image=query_image)
        output_post = instance.postprocess(output)

        out_file_path = visualize_and_save(
            query_image_path=query_image_path,
            inference_output=output_post,
            folder_to_save=f"output-{out_folder}",
            mask_weight=0.8
        )

        et = time.time()
        print(f"Saved to {out_file_path} | {st-et}")
