import torch
import os
import time
import numpy as np
import collections
import shutil
import sys

import models
from utils import load_config
import losses
from proxy_anchor_loss import ProxyAnchorLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
model = None


def main(cfg):
    global model

    # get dataloaders
    transform_tr, transform_ev = get_transforms(cfg.data)
    dl_tr = get_dataloader(
        root=os.path.join("datasets", cfg.data.dataset),
        set_type="train",
        transform=(transform_tr, transform_ev),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=False
    )

    if cfg.training.criterion == "ProxyAnchor":
        # proxy anchor loss initialization
        if not os.path.isfile(os.path.join(checkpoint_dir, "proxies_finetuned.pth")):
            dl_gen = get_dataloader(
                root=os.path.join("datasets", cfg.data.dataset),
                set_type="train",
                transform=(transform_ev,),
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                drop_last=False
            )

            model.eval()
            proxies = model.generate_proxies(dl_gen, device=cfg.training.device, proxy_cfg=cfg.proxy_anchor)

            torch.save(obj=proxies, f=os.path.join(checkpoint_dir, "proxies_finetuned.pth"))
        else:
            proxies = torch.load(os.path.join(checkpoint_dir, "proxies_finetuned.pth"))
        proxies = proxies.to(torch.float32)
        torch.cuda.empty_cache()
    else:
        proxies = None

    lr_scheduler = CosineAnnealingLR(model.optimizer, T_max=cfg.finetune.epochs, eta_min=0.0000001)

    loss_hist = collections.deque(maxlen=3)
    for epoch in range(0, cfg.finetune.epochs):
        model.train()
        for i, (image, mask) in enumerate(dl_tr):
            for k in range(1):
                t0 = time.time()

                loss = model.training_step(
                    image.to(torch.float32).to(cfg.training.device),
                    mask.to(torch.int8).to(cfg.training.device),
                    proxies=proxies
                )

                t1 = time.time()
                loss_hist.append(loss)
                print(f"EPOCH: {epoch} | ITER: {i}-{k}/{len(dl_tr)} " + \
                      f"| LOSS: {np.mean(loss_hist)} | LR: {model.optimizer.param_groups[0]['lr']} " + \
                      f"| STEP TIME: {t1 - t0}")

        if epoch % 5 == 0 and epoch != 0:
            model.eval()
            model.save_checkpoint(path=checkpoint_path, epoch=epoch)
            model.train()

        lr_scheduler.step()

    model.eval()
    model.save_checkpoint(path=checkpoint_path, epoch=epoch)
    model.train()


if __name__ == "__main__":
    # import config
    cfg = load_config("./config.yml")

    # get model
    model = models.load(cfg)
    model.to(cfg.training.device)
    model = model.to(torch.float32)
    if cfg.training.criterion != "ProxyAnchor":
        model.define_loss_function(getattr(losses, cfg.training.criterion)())
    else:
        model.define_loss_function(ProxyAnchorLoss(
            alpha=cfg.proxy_anchor.alpha,
            mrg=cfg.proxy_anchor.margin,
            nb_classes=cfg.data.nb_classes
        ))
    cfg.training.learning_rate = cfg.finetune.learning_rate
    model.define_optimizer(cfg)

    # define checkpoint
    checkpoint_dir = vars(cfg.model)
    checkpoint_dir = [f"{key}[{checkpoint_dir[key]}]" for key in checkpoint_dir]
    checkpoint_dir = "-".join(checkpoint_dir)

    checkpoint_dir = os.path.join("logs", checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "ckpt.pth")
    if os.path.isfile(checkpoint_path):
        last_epoch = model.load_checkpoint(checkpoint_path, device=cfg.training.device, load_opt=cfg.training.load_opt)
        checkpoint_path = checkpoint_path.replace("ckpt.pth", "ckpt_finetuned.pth")
    else:
        last_epoch = 0
    shutil.copy("./config.yml", os.path.join(checkpoint_dir, "config.yml"))

    # get dataloaders
    if cfg.data.folder_structure == "separate":
        from datasets.separate import get_transforms, get_dataloader
    elif cfg.data.folder_structure == "unified":
        from datasets.unified import get_transforms, get_dataloader
    elif cfg.data.folder_structure == "separate_json":
        from datasets.separate_json import get_transforms, get_dataloader
    else:
        raise NotImplementedError(
            f"cfg.data.folder_structure: {cfg.data.folder_structure} doesn't exist. Use one of separate, unified.")

    # try except keyboardinterrupt lets us save the model just before the program stops
    # so that we do not lose the model weights in the current epoch. But unfortunately,
    # something is wrong with saving, so interrupted checkpoints are not useful
    try:
        main(cfg)
    except KeyboardInterrupt:
        if model:
            model.save_checkpoint(path=checkpoint_path.replace(".pth", "_intrptd.pth"), epoch=last_epoch)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
