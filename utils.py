import numpy as np
import torch
import yaml
import json
import types
import cv2
import tqdm
import losses


def json_to_mask(json_path):
    with open(json_path) as fp:
        annots = json.load(fp)

    # import data
    img_size = annots["imageHeight"], annots["imageWidth"]

    # create mask
    mask = np.zeros(img_size, dtype=np.uint8)
    for i,_ in enumerate(annots["shapes"]):
        pts = np.asarray([annots["shapes"][i]["points"]], dtype=np.int)
        cv2.fillPoly(mask, pts=pts, color=(255, 255, 255))

    # save mask as a variant of json annotation
    mask_path = json_path.replace(".json", ".png")
    cv2.imwrite(mask_path, mask)
    return mask_path

def load_config(path_to_config_yaml="./config.yaml"):

    with open(path_to_config_yaml) as f:
        dct = yaml.safe_load(f)

    def load_object(dct):
        return types.SimpleNamespace(**dct)

    cfg = json.loads(json.dumps(dct), object_hook=load_object)

    return cfg

def to_cpu(tensor):
    return tensor.detach().clone().cpu()

def evaluate(cfg, dl_ev, model, rng=10):
    validation_loss_func = getattr(losses, cfg.training.criterion)()
    model.eval()
    validation_loss = []
    pbar = tqdm.tqdm(dl_ev)
    for i, (image, mask) in enumerate(pbar):
        if range:
            if i == rng:
                break
        with torch.no_grad():
            out = model(image.to(cfg.training.device))
            mask = mask.to(cfg.training.device)
            validation_loss.append(validation_loss_func(out, mask).detach())
        pbar.set_description(f"EVALUATING: [{i}/{len(dl_ev)}]")
    mean_validation_loss = sum(validation_loss)/len(validation_loss)
    model.train()
    return mean_validation_loss


