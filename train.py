import torch
import os
import time
import torchvision
import numpy as np
import collections
import shutil
import sys

import models
from utils import load_config, evaluate
import losses
from proxy_anchor_loss import ProxyAnchorLoss

model = None

def main(cfg):
    global model
    if not os.path.isdir("inference_examples"):
        os.makedirs("inference_examples")
        
    base_lr = cfg.training.learning_rate
    # get dataloaders
    transform_tr, transform_ev = get_transforms(cfg.data)
    dl_tr = get_dataloader(
        root=os.path.join("datasets", cfg.data.dataset),
        set_type="train",
        transform=transform_tr,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    dl_ev = get_dataloader(
        root=os.path.join("datasets", cfg.data.dataset),
        set_type="val",
        transform=transform_ev,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer,
                                                           mode='min',
                                                           factor=0.2,
                                                           patience=2,
                                                           verbose=True,
                                                           threshold=0.01,
                                                           min_lr=0.0000005)

    if cfg.training.criterion == "ProxyAnchor":
        # proxy anchor loss initialization
        dl_gen = get_dataloader(
            root=os.path.join("datasets", cfg.data.dataset),
            set_type="train",
            transform=transform_ev,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            drop_last=False
        )

        model.eval()
        proxies = model.generate_proxies(dl_gen, device=cfg.training.device, proxy_anchor_cfg=cfg.proxy_anchor)

    else:
        proxies = None

    model.train()
    loss_hist = collections.deque(maxlen=30)
    for epoch in range(last_epoch, cfg.training.epochs):
        epoch_loss = []
        for i, (image, mask) in enumerate(dl_tr):
            for k in range(1):
                t0 = time.time()
                loss = model.training_step(
                    image.to(cfg.training.device),
                    mask.to(cfg.training.device),
                    proxies=proxies
                )
                t1 = time.time()
                loss_hist.append(loss)
                print(f"EPOCH: {epoch} | ITER: {i}-{k}/{len(dl_tr)} " + \
                      f"| LOSS: {np.mean(loss_hist)} | LR: {model.optimizer.param_groups[0]['lr']} " + \
                      f"| STEP TIME: {t1-t0}")

        model.eval()
        pred = model.infer_sigmoid(image.to(cfg.training.device))
        [torchvision.utils.save_image(pred[:, i].unsqueeze(1).cpu(), f"./inference_examples/pred_{i}.png") for i in range(pred.shape[1])]
        
        mask = mask.argmax(1).unsqueeze(1).to(torch.float32).cpu()
        torchvision.utils.save_image(mask, "./inference_examples/mask.png")
        model.train()
        
        epoch_loss.append(loss)
        
        val_loss = evaluate(cfg, dl_ev, model, rng=1000)
        print(f"VALIDATION LOSS: {val_loss}")
        
        scheduler.step(np.mean(epoch_loss))
        print(model.optimizer.param_groups[0]["lr"], "<=", model.optimizer.param_groups[0]["lr"] <= 0.0000005)
        if model.optimizer.param_groups[0]["lr"] <= 0.0000005:
            for opt in model.optimizer.param_groups:
                opt['lr'] = (base_lr + opt['lr'])/2
                base_lr = opt['lr']
        
        model.save_checkpoint(path=checkpoint_path, epoch=epoch)
        # visualize(model, dl_ev, device=cfg.training.device)

if __name__ == "__main__":
    # import config
    cfg = load_config("./config.yml")

    # get model
    model = models.load(cfg)
    model.to(cfg.training.device)
    if cfg.training.criterion != "ProxyAnchor":
        model.define_loss_function(getattr(losses, cfg.training.criterion)())
    else:
        model.define_loss_function(ProxyAnchorLoss(
            alpha=cfg.proxy_anchor.alpha,
            mrg=cfg.proxy_anchor.margin,
            nb_classes=cfg.data.nb_classes
        ))
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
        raise NotImplementedError(f"cfg.data.folder_structure: {cfg.data.folder_structure} doesn't exist. Use one of separate, unified.")

    # try except keyboardinterrupt lets us save the model just before the program stops
    # so that we do not lose the model weights in the current epoch.
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