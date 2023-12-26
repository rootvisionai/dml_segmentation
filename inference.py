import os
import time

import torch
import torchvision
import numpy as np
import cv2

import models
from utils import load_config



class Inference:
    def __init__(self, cfg, checkpoint_path):
        self.cfg = cfg
        # get model
        self.model = models.load(cfg)
        self.model.to(cfg.inference.device)
        self.model.eval()

        last_epoch = self.model.load_checkpoint(checkpoint_path, train=False, device=cfg.training.device)

        print(
            f"Loaded model: {checkpoint_path}\n",
            f"Last epoch: {last_epoch}"
        )
        self.colors = [
            (0, 0, 0),
            (255, 0, 255)
        ]

    def preprocess(self, image_paths, size=(256, 256)):
        images = []
        out_sizes = []
        originals = []
        for path in image_paths:
            image = cv2.imread(path)  # load
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # to rgb
            image = torch.from_numpy(image)  # to uint8 tensor
            out_sizes.append(tuple(image.shape[0:2]))

            image = image.permute(2, 0, 1).unsqueeze(0) / 255  # to [0,1] range bs 1 float32 tensor
            image = torchvision.transforms.functional.crop(image, top=0, left=450, height=2048, width=1480) # x_min=450, y_min=0, x_max=1930, y_max=2048
            originals.append(torch.nn.functional.interpolate(image, size=self.cfg.inference.out_size).squeeze(0))

            image = torch.nn.functional.interpolate(image, size=size, mode="nearest")
            images.append(image[0])

        images = torch.stack(images, dim=0)
        return originals, images, out_sizes

    def load_images(self, input_dir):
        self.image_paths = [os.path.join(input_dir, elm) for elm in os.listdir(input_dir)]
        originals, inputs, out_sizes = self.preprocess(self.image_paths, size=(cfg.data.input_size, cfg.data.input_size))
        return originals, inputs, out_sizes

    def process(self, inputs):
        print(f"Running inference on images: {inputs.shape}")
        inputs = inputs.to(cfg.inference.device)
        preds, confs = self.model.infer(inputs)
        return preds, confs

    def postprocess(self, preds, out_size):
        processed = []
        for bs in range(preds.shape[0]):
            intermediate = []
            for ch in range(self.cfg.model.num_classes):
                pred = preds[bs].unsqueeze(0)
                pred = torch.nn.functional.interpolate(pred, size=out_size, mode="nearest")
                p = torch.zeros_like(pred)
                coords = torch.where(pred == ch)
                p[:, :, coords[2], coords[3]] = 1.
                intermediate.append(p.squeeze(0).squeeze(0))
            intermediate = torch.stack(intermediate, dim=0)
            processed.append(intermediate)
        processed = torch.stack(processed, dim=0)
        return processed

    def pipeline(self, input_dir, output_dir):
        originals, inputs, out_sizes = self.load_images(input_dir)
        st = time.time()
        preds, confs = self.process(inputs)
        et = time.time()
        print(f"INFERENCE TIME: {et-st}")
        preds = self.postprocess(preds, self.cfg.inference.out_size)
        rendered_images = []

        for bs in range(preds.shape[0]):
            original = (255*originals[bs].permute(1, 2, 0).numpy()).astype(np.uint8)

            for ch in range(1, preds.shape[1]):
                p = (255*preds[bs, ch].cpu().numpy()).astype(np.uint8)
                rendered = self.render_mask(original, p, ch)
                polygon_pts = instance.convert_to_polygon(p)
                bbox_pts = instance.convert_to_bbox(polygon_pts)
                rendered = instance.render_bbox(rendered, bbox_pts)
                if len(polygon_pts)>0:
                    save_dir = os.path.join(output_dir, f"{bs}_{ch}.jpg")
                    cv2.imwrite(save_dir, rendered)

            rendered_images.append(rendered)
        return rendered_images

    def render_mask(self, input_image, pred, cnt):
        color = np.array(self.colors[cnt], dtype='uint8')
        masked_img = np.where(pred[..., None] > 0.9*255, color, input_image)
        out = cv2.addWeighted(input_image, 0.5, masked_img, 0.5, 0)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out

    def convert_to_polygon(self, mask):
        coords = np.column_stack(np.where(mask > 0.9*255))
        return coords

    def convert_to_bbox(self, polygon_points):
        if len(polygon_points) > 0:
            ymin = polygon_points[:, 0].min()
            xmin = polygon_points[:, 1].min()
            ymax = polygon_points[:, 0].max()
            xmax = polygon_points[:, 1].max()
        else:
            ymin = xmin = ymax = xmax = 0
        return {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax
        }

    def render_bbox(self, img, bbox):
        cv2.rectangle(img, (bbox["xmin"], bbox["ymin"]), (bbox["xmax"], bbox["ymax"]), (0, 255, 0), 1)
        return img

    def save_with_mask(self, input_dir, output_dir):
        for bs in range(self.preds.shape[0]):
            for ch in range(cfg.model.num_classes):
                p = torch.zeros_like(self.originals[bs].unsqueeze(0))
                pred = self.preds[bs].unsqueeze(0)
                pred = torch.nn.functional.interpolate(pred, size=self.out_sizes[bs], mode="nearest")
                coords_pos = torch.where(pred == ch)
                coords_neg = torch.where(pred != ch)
                p[:, :, coords_pos[2], coords_pos[3]] = self.originals[bs][:, coords_pos[2], coords_pos[3]]
                p[:, :, coords_neg[2], coords_neg[3]] = self.originals[bs][:, coords_neg[2], coords_neg[3]] * 0.2
                save_dir = self.image_paths[bs].replace(input_dir, output_dir)
                save_dir = save_dir.replace(".jpg", f"_{bs}_{ch}.png")
                torchvision.utils.save_image(p, save_dir)


if __name__ == "__main__":
    checkpoint_dir = "arch[UNetResnet]-num_classes[2]-in_channels[3]-backbone[resnet50]-output_stride[16]-freeze_bn[False]-freeze_backbone[False]"

    cfg = load_config(
        "config.yml"
    )

    checkpoint_path = os.path.join(
        "checkpoints",
        f"{checkpoint_dir}",
        "ckpt.pth"
    )

    instance = Inference(cfg, checkpoint_path=checkpoint_path)
    rendered = instance.pipeline(
        input_dir="test_input",
        output_dir="test_output"
    )



