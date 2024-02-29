import os
import shutil
import json
import logging
import hashlib
import copy
from tqdm import tqdm
import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import supervision as sv
from sklearn.cluster import KMeans

from utils import load_config
from sam.models import load_model
from sam.utils_sam import (generate_embedding, load_dataset, view_clusters,
                           Mask2Polygon, create_rectangular_crop, fiter_target_mask,
                           get_images_and_annots, get_ooi)

import warnings
warnings.filterwarnings('ignore')


def calculate_data_md5(annotations):
    s = ""
    for annot in annotations:
        with open(annot, 'r') as f:
            s += f.read()
    md5_value = hashlib.md5(s.encode('utf-8')).hexdigest()

    return md5_value


def get_data(cfg):
    original_folder = os.path.join(cfg.data.root_path, "data", "original")
    # check if data is in correct folder
    assert os.path.exists(original_folder), f"Please put your data in {cfg.data.root_path}/data/original"
    assert len(os.listdir(original_folder)) > 0, f"There is no data in {cfg.data.root_path}/data/original"

    # get original images and annotations
    img_list, annot_list = get_images_and_annots(img_folder=original_folder)

    return img_list, annot_list


def check_reusability(cfg, annot_list):
    sam_name = cfg.sam_model_name
    md5_value = calculate_data_md5(annot_list)

    aux_dir_name = os.path.join(cfg.data.root_path, "data", "auxiliary")
    hash_path = os.path.join(aux_dir_name, "hash.txt")
    if os.path.exists(hash_path):
        with open(hash_path) as file:
            lines = file.read().splitlines()
            hash_sam_name = lines[0]
            hash_md5_value = lines[1]
        if sam_name == hash_sam_name and md5_value == hash_md5_value:
            return True
        else:
            shutil.rmtree(aux_dir_name)

    else:
        if os.path.exists(aux_dir_name):
            shutil.rmtree(aux_dir_name)
        else:
            os.makedirs(aux_dir_name, exist_ok=True)

        return False


def sam_inference(cfg, img_list, annot_list, device):

    logging.info("Starting SAM inference...")
    sam = sam_model_registry[cfg.sam_model_name](checkpoint=cfg.sam_model_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=cfg.min_mask_region_area)

    aux_path = os.path.join(cfg.data.root_path, "images", "data", "auxiliary")
    polygons_results = []
    for i, (img_path, annot_path) in tqdm(enumerate(zip(img_list, annot_list))):

        # reading image
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # inference
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sam_output = mask_generator.generate(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # visualize sam
        if cfg.visualise:
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            detections = sv.Detections.from_sam(sam_result=sam_output)
            annotated_image = mask_annotator.annotate(scene=copy.deepcopy(img), detections=detections)
            out_path = os.path.join(aux_path, "sam_visualized")
            os.makedirs(out_path, exist_ok=True)
            cv2.imwrite(os.path.join(out_path, os.path.basename(img_path)), annotated_image)

        # create raw masks
        create_rectangular_crop(rgb_image=img, results=sam_output,
                                output_dir=os.path.join(aux_path, "masks_raw"), index=i)

        # open original annotation
        with open(annot_path, 'r') as file:
            original_json = json.load(file)

        # get masks of the object of interest (ooi)
        ooi_masks = get_ooi(json_file=original_json, width=width, height=height)
        sam_output_filtered = fiter_target_mask(masks_roi=ooi_masks, sam_output=sam_output,
                                                threshold=cfg.data.threshold_orig_overlap)
        # create masks with filtered ooi
        create_rectangular_crop(rgb_image=img, results=sam_output_filtered,
                                output_dir=os.path.join(aux_path, "masks_filtered", "crops"),
                                index=i)

        masks = [result["segmentation"] for result in sam_output_filtered]
        mask2polygon_converter = Mask2Polygon(min_area=cfg.data.min_area, epsilon=cfg.data.epsilon)
        poly_res = []
        for mask in masks:
            res = mask2polygon_converter.rle2polygon(mask)
            poly_res.append(res)

        polygons_results.append(poly_res)

    np.save(os.path.join(aux_path, "polygons.npy"), np.array(polygons_results, dtype=object), allow_pickle=True)
    logging.info("SAM inference completed")


def get_embeddings(cfg, device):
    logging.info("Getting crops embeddings...")
    ckpt_name = (f"arch[{cfg.model.arch}]-backbone[{cfg.model.backbone}]-emb_size-[{cfg.model.embedding_size}]-"
                 f"input_size[{cfg.data.preprocessing.input_size}]")
    aux_path = os.path.join(cfg.data.root_path, "images", "data", "auxiliary")
    cfg.checkpoint_dir = os.path.join(aux_path, "logs", ckpt_name)

    if os.path.isfile(os.path.join(cfg.checkpoint_dir, "collection.pth")):
        logging.info("Skipping embeddings part, cached data was found")
    else:
        logging.info("Getting crops embeddings...")
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        model = load_model(cfg)
        model.to(device)
        data_path = os.path.join(aux_path, "masks_filtered")
        dl_gen = load_dataset(cfg, data_path)
        generate_embedding(cfg, model, dl_gen)
        logging.info("Embeddings are generated")


def get_clusters(cfg, img_list, annot_list):
    logging.info("Starting clustering process...")
    collection = torch.load(os.path.join(cfg.checkpoint_dir, "collection.pth"))
    feat = np.array(collection["embeddings"])

    last_run = 0
    dir_path = os.path.join(cfg.data.root_path, "runs")
    files_dir = [
        f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
    ]
    for subdir in files_dir:
        run = subdir.split("exp")[-1]
        if int(run) > last_run:
            last_run = int(run)

    result_dir = os.path.join(dir_path, "exp" + str(last_run + 1))
    os.makedirs(result_dir, exist_ok=True)

    aux_path = os.path.join(cfg.data.root_path, "images", "data", "auxiliary")
    polygons_results = np.load(os.path.join(aux_path, "polygons.npy"), allow_pickle=True)

    # paths for result annotations
    annot_list_modified = [os.path.join(result_dir, os.path.basename(annot)) for annot in annot_list]

    # cluster feature vectors
    kmeans = KMeans(n_clusters=cfg.data.n_clusters, random_state=0)
    kmeans.fit(feat)

    logging.info("Clustering is done")

    counter = 0
    for i, (results, annot_path) in tqdm(enumerate(zip(polygons_results, annot_list))):
        # copy image in result folder
        shutil.copy(img_list[i], os.path.join(result_dir, os.path.basename(img_list[i])))

        with open(annot_path, 'r') as file:
            json_modified = json.load(file)

        for polygons in results:
            label = kmeans.labels_[counter]
            counter += 1
            for polygon in polygons:
                json_modified["shapes"].append({'flags': {}, 'group_id': None, 'description': "", 'shape_type': 'polygon',
                                                'mask': None, 'label': f"class_{str(label)}", 'points': polygon})
        with open(annot_list_modified[i], 'w', encoding='utf-8') as f:
            json.dump(json_modified, f, ensure_ascii=False, indent=4)

    # holds the cluster id and the images { id: [images] }
    groups = {}
    for file, cluster in zip(collection["img_paths"], kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    cluster_output_dir = os.path.join(cfg.data.root_path, "images", "clusters")
    for i, label in enumerate(np.unique(kmeans.labels_)):
        view_clusters(groups=groups, cluster=label, output_dir=cluster_output_dir, cluster_num=i)

    logging.info(f"Result annotations are created. Find them under {result_dir}")


if __name__ == "__main__":
    # load config
    config = load_config("./config.yml")
    config = config.sam

    # set device
    dvc = 'cuda' if torch.cuda.is_available() else 'cpu'

    images, annotations = get_data(cfg=config)
    if check_reusability(cfg=config, annot_list=annotations):
        logging.info("SAM inference is skipped, cached data was found")
    else:
        sam_inference(cfg=config, img_list=images, annot_list=annotations, device=dvc)

    get_embeddings(cfg=config, device=dvc)
    get_clusters(cfg=config, img_list=images, annot_list=annotations)
