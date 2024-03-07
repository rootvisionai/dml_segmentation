import os
import shutil
import json
import copy
from tqdm import tqdm
import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import supervision as sv

import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.INFO)

from utils import load_config
from sam.utils_sam import (calculate_data_md5, fill_hash_file,
                           Mask2Polygon, create_rectangular_crop, fiter_target_mask,
                           get_images_and_annots, get_ooi, get_embeddings, get_clusters)


def get_data(cfg):
    original_folder = os.path.join(cfg.data.root_path, "data", "original")
    # check if data is in correct folder
    assert os.path.exists(original_folder), f"Please put your data in {cfg.data.root_path}/data/original"
    assert len(os.listdir(original_folder)) > 0, f"There is no data in {cfg.data.root_path}/data/original"

    # get original images and annotations
    img_list, annot_list = get_images_and_annots(img_folder=original_folder)

    return img_list, annot_list


def check_reusability(cfg, md5):
    sam_name = cfg.sam_model_name
    min_mask_region_area = str(cfg.data.min_mask_region_area)
    threshold_orig_overlap = str(cfg.data.threshold_orig_overlap)
    min_polygon_area = str(cfg.data.min_polygon_area)
    epsilon = str(cfg.data.epsilon)

    aux_dir_name = os.path.join(cfg.data.root_path, "data", "auxiliary")
    hash_path = os.path.join(aux_dir_name, "hash_and_hyps.txt")
    if os.path.exists(hash_path):
        with open(hash_path) as file:
            lines = file.read().splitlines()
            hash_sam_name = lines[0].split(" ")[-1]
            hash_min_mask_region_area = lines[1].split(" ")[-1]
            hash_threshold_orig_overlap = lines[2].split(" ")[-1]
            hash_min_polygon_area = lines[3].split(" ")[-1]
            hash_epsilon = lines[4].split(" ")[-1]
            hash_md5_value = lines[5].split(" ")[-1]
        if (sam_name == hash_sam_name and min_mask_region_area == hash_min_mask_region_area
            and threshold_orig_overlap == hash_threshold_orig_overlap and min_polygon_area == hash_min_polygon_area
                and epsilon == hash_epsilon and md5 == hash_md5_value):

            return True
        else:
            shutil.rmtree(aux_dir_name)
            return False

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
    mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=cfg.data.min_mask_region_area)

    aux_path = os.path.join(cfg.data.root_path, "data", "auxiliary")
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

        logging.info("Creating raw masks")
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
        logging.info("Creating filtered masks")
        create_rectangular_crop(rgb_image=img, results=sam_output_filtered,
                                output_dir=os.path.join(aux_path, "masks_filtered", "crops"),
                                index=i)

        masks = [result["segmentation"] for result in sam_output_filtered]
        mask2polygon_converter = Mask2Polygon(min_area=cfg.data.min_polygon_area, epsilon=cfg.data.epsilon)
        poly_res = []
        for mask in masks:
            res = mask2polygon_converter.rle2polygon(mask)
            poly_res.append(res)

        polygons_results.append(poly_res)

    np.save(os.path.join(aux_path, "polygons.npy"), np.array(polygons_results, dtype=object), allow_pickle=True)
    logging.info("SAM inference completed")


if __name__ == "__main__":
    # load config
    config = load_config("./config.yml")
    config = config.sam

    # set device
    dvc = 'cuda' if torch.cuda.is_available() else 'cpu'

    images, annotations = get_data(cfg=config)
    # calculates md5 for all jsons
    md5_value = calculate_data_md5(annotations)
    if check_reusability(cfg=config, md5=md5_value):
        logging.info("SAM inference is skipped, cached data was found")
    else:
        fill_hash_file(cfg=config, md5=md5_value)
        sam_inference(cfg=config, img_list=images, annot_list=annotations, device=dvc)

    get_embeddings(cfg=config, device=dvc)
    get_clusters(cfg=config, img_list=images, annot_list=annotations)
