import os
import shutil
import json
import logging
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


if __name__ == "__main__":
    # load config
    cfg = load_config("./config.yml")
    cfg = cfg.sam

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    original_folder = os.path.join(cfg.data.root_path, "images", "original")
    # check if data is in correct folder
    assert os.path.exists(original_folder), f"Please put your data in {cfg.data.root_path}/images/original"
    assert len(os.listdir(original_folder)) > 0, f"There is no data in {cfg.data.root_path}/images/original"

    # get original images and annotations
    img_list, annot_list = get_images_and_annots(config=cfg, img_folder=original_folder)

    # create result dir
    result_dir = os.path.join(cfg.data.root_path, "images", "result")
    os.makedirs(result_dir, exist_ok=True)

    # paths for result annotations
    annot_list_modified = [os.path.join(result_dir, os.path.basename(annot)) for annot in annot_list]

    # polygons for all images and all their crops that we got after sam inference
    polygons_results = []

    logging.info("Starting SAM inference")

    sam_inference_path = os.path.join(cfg.data.root_path, "images", "runs", "sam_inference")
    os.makedirs(sam_inference_path, exist_ok=True)
    sam_exp_list = [s for s in os.listdir(sam_inference_path) if "exp" in s]
    for i, (img_path, annot_path) in tqdm(enumerate(zip(img_list, annot_list))):

        # copy image in result folder
        shutil.copy(img_path, os.path.join(result_dir, os.path.basename(img_path)))

        # reading image
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # open original annotation
        with open(annot_path, 'r') as file:
            original_json = json.load(file)

        # get masks of the object of interest
        ooi_masks = get_ooi(json_file=original_json, width=width, height=height)

        # save inference results in npy files if cuda not available or for debugging purposes
        sam_output_path = None
        if device != "cuda" or cfg.debug:
            os.makedirs(os.path.join(cfg.data.root_path, "tmp"), exist_ok=True)
            sam_output_name = os.path.basename(annot_path).replace(".json", "")
            sam_output_path = os.path.join(cfg.data.root_path, "tmp", f"{sam_output_name}.npy")

        if os.path.exists(sam_output_path):
            sam_output = np.load(sam_output_path, allow_pickle=True)
        else:
            # inference
            sam = sam_model_registry[cfg.sam_model_name](checkpoint=cfg.sam_model_path)
            sam.to(device=device)
            mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=cfg.model.min_mask_region_area)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sam_output = mask_generator.generate(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if device != "cuda" or cfg.debug:
                np.save(sam_output_path, np.array(sam_output, dtype=object), allow_pickle=True)

        # visualize sam
        if cfg.visualise:
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            detections = sv.Detections.from_sam(sam_result=sam_output)
            annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
            out_path = os.path.join(cfg.data.root_path, "images", "sam_visualized")
            os.makedirs(out_path, exist_ok=True)
            cv2.imwrite(os.path.join(out_path, os.path.basename(img_path)), annotated_image)

        create_rectangular_crop(rgb_image=img, results=sam_output,
                                output_dir=os.path.join(cfg.data.root_path, "images", "masks_raw"), index=i)

        a_post_tmp = fiter_target_mask(masks_roi=ooi_masks, sam_output=sam_output,
                                       threshold=cfg.data.threshold_orig_overlap)
        masks = [result["segmentation"] for result in a_post_tmp]
        mask2polygon_converter = Mask2Polygon(min_area=cfg.data.min_area, epsilon=cfg.data.epsilon)
        poly_res = []
        for mask in masks:
            res = mask2polygon_converter.rle2polygon(mask)
            poly_res.append(res)

        polygons_results.append(poly_res)

        create_rectangular_crop(rgb_image=img, results=a_post_tmp,
                                output_dir=os.path.join(cfg.data.root_path, "images", "masks_filtered/crops"), index=i)

    np.save("./tmp_polygons.npy", np.array(polygons_results, dtype=object), allow_pickle=True)
    # -----------------------------

    polygons_results = np.load("./tmp_polygons.npy", allow_pickle=True)

    ckpt_name = (f"arch[{cfg.model.arch}]-backbone[{cfg.model.backbone}]-emb_size-[{cfg.model.embedding_size}]-"
                 f"input_size[{cfg.data.preprocessing.input_size}]")
    cfg.checkpoint_dir = os.path.join(cfg.data.root_path, "logs", ckpt_name)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    model = load_model(cfg)
    model.to(device)

    dl_gen = load_dataset(cfg)

    if os.path.isfile(os.path.join(cfg.checkpoint_dir, "collection.pth")):
        collection = torch.load(os.path.join(cfg.checkpoint_dir, "collection.pth"))
    else:
        collection = generate_embedding(cfg, model, dl_gen)

    feat = np.array(collection["embeddings"])

    os.makedirs(os.path.join(cfg.data.root_path, "images", "runs", "result"), exist_ok=True)


    # cluster feature vectors
    kmeans = KMeans(n_clusters=cfg.data.n_clusters, random_state=0)
    kmeans.fit(feat)

    counter = 0
    for i, (results, annot_path) in tqdm(enumerate(zip(polygons_results, annot_list))):
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
