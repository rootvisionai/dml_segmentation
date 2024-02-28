import torch
import tqdm
import os
import torchvision
from imantics import Polygons
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .augmentations import TransformEvaluate


def generate_embedding(cfg, model, dl_gen):
    model.eval()
    model = model.to(cfg.training.device)
    pbar = tqdm.tqdm(enumerate(dl_gen))
    img_paths = []
    # labels_int = []
    # labels_str = []
    for batch_id, batch in pbar:
        with torch.no_grad():
            emb = model(batch["image"].to(cfg.training.device))
        # lint = batch["label_int"]
        # lstr = batch["label_str"]
        i_paths = batch["img_path"]
        if batch_id == 0:
            embeddings = emb.cpu()
            # labels_int = lint.cpu()
        else:
            embeddings = torch.cat([embeddings, emb.cpu()], dim=0)
            # labels_int = torch.cat([labels_int, lint.cpu()], dim=0)
        # labels_str = labels_str + lstr
        img_paths += i_paths
        pbar.set_description(f"GENERATING EMBEDDING COLLECTION: [{batch_id}/{len(dl_gen)}]")

    # label_map = []
    # for elm in labels_str:
    #     if elm not in label_map:
    #         label_map.append(elm)

    collection = {
        "embeddings": embeddings.detach(),
        "img_paths": img_paths
        # "labels_str": labels_str,
        # "labels_int": labels_int,
        # "label_map" : label_map
    }

    torch.save(
        collection,
        os.path.join(cfg.checkpoint_dir, "collection.pth")
    )

    return collection


def img_load(path):
    im = Image.open(path).convert('RGB')
    return im


class ImageFolder(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        record = self.samples[index]
        img_path = record[0]
        image = img_load(img_path)
        label_int = record[1]
        label_str = self.classes[label_int]
        if self.transform is not None:
            image = self.transform(image)

        sample = {
            "image": image,
            "label_int": label_int,
            "label_str": label_str,
            "img_path": img_path
        }
        return sample

    def nb_classes(self):
        return len(self.classes)


def load_dataset(cfg):

    data_path = os.path.join(cfg.data.root_path, "images", "masks_filtered")

    ds = ImageFolder(
        root=data_path,
        transform=TransformEvaluate(cfg)
    )

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        collate_fn=collater,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        drop_last=False,
        pin_memory=True
    )

    return dl


def collater(data):
    # Function to pull single image
    # and put to batch. This is for Pytorch dataloader
    # to load input batches.
    keys = list(data[0].keys())

    out = {}
    for key in keys:
        data_piece = [s[key] for s in data]
        if key == "image":
            data_piece = torch.stack(data_piece, dim=0)
        elif key in ["score", "label_int", "confidence"]:
            data_piece = torch.tensor(data_piece)
        else:
            pass
        out[key] = data_piece

    return out


class Mask2Polygon:
    def __init__(self, min_area=10, epsilon=0.001):
        self.min_area = min_area  # min area of contour
        self.epsilon = epsilon  # contour point density coefficient

    @staticmethod
    def is_clockwise(contour):
        value = 0
        num = len(contour)
        for i, point in enumerate(contour):
            p1 = contour[i]
            if i < num - 1:
                p2 = contour[i + 1]
            else:
                p2 = contour[0]
            value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1])
        return value < 0

    @staticmethod
    def get_merge_point_idx(contour1, contour2):
        idx1 = 0
        idx2 = 0
        distance_min = -1
        for i, p1 in enumerate(contour1):
            for j, p2 in enumerate(contour2):
                distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2)
                if distance_min < 0:
                    distance_min = distance
                    idx1 = i
                    idx2 = j
                elif distance < distance_min:
                    distance_min = distance
                    idx1 = i
                    idx2 = j
        return idx1, idx2

    @staticmethod
    def merge_contours(contour1, contour2, idx1, idx2):
        contour = []
        for i in list(range(0, idx1 + 1)):
            contour.append(contour1[i])
        for i in list(range(idx2, len(contour2))):
            contour.append(contour2[i])
        for i in list(range(0, idx2 + 1)):
            contour.append(contour2[i])
        for i in list(range(idx1, len(contour1))):
            contour.append(contour1[i])
        contour = np.array(contour)
        return contour

    def merge_with_parent(self, contour_parent, contour):
        if not self.is_clockwise(contour_parent):
            contour_parent = contour_parent[::-1]
        if self.is_clockwise(contour):
            contour = contour[::-1]
        idx1, idx2 = self.get_merge_point_idx(contour_parent, contour)
        return self.merge_contours(contour_parent, contour, idx1, idx2)

    def mask2polygon(self, mask):
        mask = mask.astype(np.uint8)
        # mask = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        # contours, hierarchies = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
        raw_contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        contours_approx = []
        for contour in raw_contours:
            epsilon = self.epsilon * cv2.arcLength(contour, True)
            contour_approx = cv2.approxPolyDP(contour, epsilon, True)
            if cv2.contourArea(contour_approx) >= self.min_area:
                contours_approx.append(contour_approx)
        del raw_contours

        contours_parent = []
        for i, contour in enumerate(contours_approx):
            parent_idx = hierarchies[0][i][3]
            if parent_idx < 0 and len(contour) >= 3:
                contours_parent.append(contour)
            else:
                contours_parent.append([])

        # merge if there is hole inside
        for i, contour in enumerate(contours_approx):
            parent_idx = hierarchies[0][i][3]
            if parent_idx >= 0 and len(contour) >= 3:
                contour_parent = contours_parent[parent_idx]
                if len(contour_parent) == 0:
                    continue
                contours_parent[parent_idx] = self.merge_with_parent(contour_parent, contour)
        del contours_approx

        polygons = [np.squeeze(contour, axis=1).tolist() for contour in contours_parent if len(contour) > 0]

        return polygons

    def rle2polygon(self, mask):
        polygons = self.mask2polygon(mask)
        return polygons


def get_images_and_annots(config, img_folder):
    img_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = [os.path.join(config.data.root_path, "images", "original", file) for file in os.listdir(img_folder)
                if file.lower().endswith(img_ext)]
    annotations = []
    for image in images:
        ext = "." + image.split(".")[-1]
        annot = image.replace(ext, ".json")
        assert os.path.exists(annot), f"annotation {annot} doesn't exist!"
        annotations.append(annot)

    return images, annotations


def get_ooi(json_file, width, height):
    annotation = json_file["shapes"]
    polygons = [[shape["points"] for shape in annotation]]
    masks = [Polygons(polygon).mask(width=width, height=height).array for polygon in polygons]

    return masks


def create_rectangular_crop(rgb_image, results, output_dir, index):
    os.makedirs(output_dir, exist_ok=True)
    for i, result in tqdm.tqdm(enumerate(results)):
        mask_array = result["segmentation"]
        bbox = result["bbox"]

        left, top, right, bottom = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

        # Create rectangular crop of the original image
        crop = rgb_image[top:bottom+1, left:right+1].copy()

        # Replace pixels with 0 in the mask with black
        crop[mask_array[top:bottom+1, left:right+1] == 0] = [0, 0, 0]

        # Save the resulting crop

        output_path = os.path.join(output_dir, f"mask_{'{:03d}'.format(index)}_{'{:04d}'.format(i)}.jpg")
        cv2.imwrite(output_path, crop)


def fiter_target_mask(masks_roi, sam_output, threshold=0.1):
    masks_background = [mask["segmentation"] for mask in sam_output]
    masks_sam_filtered = []
    for i, mask_bckg in enumerate(masks_background):
        flag = 1
        for roi in masks_roi:
            img_bwa = mask_bckg & roi
            img_bwo = mask_bckg | roi
            iou = np.sum(img_bwa) / np.sum(img_bwo)
            if iou > threshold:
                flag = 0
                break
            elif iou > 0.0:
                mask_bckg[img_bwa[::] == 1] = 0
        if flag:
            masks_sam_filtered.append(sam_output[i])

    return masks_sam_filtered


def view_clusters(groups, cluster, output_dir, cluster_num):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(30, 30))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Add spacing between subplots
    plt.suptitle(f'Cluster {cluster_num}', fontsize=30)  # Add a title to the cluster

    # gets the list of filenames for a cluster
    files = groups[cluster]
    num_files = min(len(files), 30)  # Limit to maximum 30 files
    num_rows = (num_files - 1) // 5 + 1  # Calculate number of rows based on number of files

    # plot each image in the cluster
    for index, file_path in enumerate(files[:30]):
        plt.subplot(num_rows, 5, index + 1)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"cluster_{str(cluster_num)}.jpg"))
