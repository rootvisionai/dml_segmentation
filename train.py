import torch
import os
import time
import numpy as np
import collections
import shutil
import sys

import torchvision.utils

import models
from utils import load_config, evaluate_with_knn
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
        transform=(transform_tr, transform_ev),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    dl_ev = get_dataloader(
        root=os.path.join("datasets", cfg.data.dataset),
        set_type="val",
        transform=(transform_ev, ),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer,
                                                           mode='min',
                                                           factor=0.2,
                                                           patience=2,
                                                           verbose=True,
                                                           threshold=0.1,
                                                           min_lr=0.0000005)

    if cfg.training.criterion == "ProxyAnchor":
        # proxy anchor loss initialization
        if not os.path.isfile(os.path.join(checkpoint_dir, "proxies.pth")):
            dl_gen = get_dataloader(
                root=os.path.join("datasets", cfg.data.dataset),
                set_type="val",
                transform=(transform_ev, ),
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                drop_last=False
            )

            model.eval()
            # model = model.to(torch.float32)
            proxies = model.generate_proxies(dl_gen, device=cfg.training.device, proxy_cfg=cfg.proxy_anchor)

            torch.save(obj=proxies, f=os.path.join(checkpoint_dir, "proxies.pth"))
        else:
            proxies = torch.load(os.path.join(checkpoint_dir, "proxies.pth"))
        proxies = proxies.to(torch.float32)

    else:
        proxies = None

    if not os.path.isfile(os.path.join(checkpoint_dir, "validation_logs.txt")):
        with open(os.path.join(checkpoint_dir, "validation_logs.txt"), "a+") as fp:
            fp.write("Epoch,Iter,GeneralCounter,MaxPrecision,Precision@1,Precision@2,Precision@4,Precision@8\n")

    # model = model.to(torch.float16)

    loss_hist = collections.deque(maxlen=30)
    scheduler_loss = collections.deque(maxlen=500)
    cnt = 0
    for epoch in range(last_epoch, cfg.training.epochs):
        model.train()
        for i, (image, mask) in enumerate(dl_tr):
            for k in range(1):
                t0 = time.time()

                # image = torch.cat([image, image_], dim=0)
                # mask = torch.cat([mask, mask_], dim=0)
                torchvision.utils.save_image(mask.sum(dim=1).unsqueeze(1)/81, fp="debug_masks.png")

                # proxies, classes = model.generate_temp_proxy(
                #     image.to(torch.float16).to(cfg.training.device),
                #     mask.to(torch.int8).to(cfg.training.device),
                #     cfg.proxy_anchor,
                #     device=cfg.training.device
                # )
                # mask = mask[:, classes, :, :]

                loss = model.training_step(
                    image.to(torch.float32).to(cfg.training.device),
                    mask.to(torch.int8).to(cfg.training.device),
                    proxies=proxies
                )
                t1 = time.time()
                loss_hist.append(loss)
                print(f"EPOCH: {epoch} | ITER: {i}-{k}/{len(dl_tr)} " + \
                      f"| LOSS: {np.mean(loss_hist)} | LR: {model.optimizer.param_groups[0]['lr']} " + \
                      f"| STEP TIME: {t1-t0}")

            if cnt % 1000 == 0 and cnt != 0:
                model.eval()
                precisions, max_precision = evaluate_with_knn(cfg, dl_ev, model, exclude_background=True, rng=100)
                precisions = ["{:.2f}".format(elm) for elm in precisions]
                with open(os.path.join(checkpoint_dir, "validation_logs.txt"), "a+") as fp:
                    fp.write(f"{epoch},{i},{cnt},{max_precision},{','.join(precisions)}\n")
                model.train()

            if (cnt+1) % 500 == 0 and cnt != 0:
                model.save_checkpoint(path=checkpoint_path, epoch=epoch)

            # del image, mask; torch.cuda.empty_cache()
            cnt += 1

        scheduler_loss.append(loss)
        scheduler.step(np.mean(scheduler_loss))
        print(model.optimizer.param_groups[0]["lr"], "<=", model.optimizer.param_groups[0]["lr"] <= 0.0000005)
        if model.optimizer.param_groups[0]["lr"] <= 0.0000005:
            for opt in model.optimizer.param_groups:
                opt['lr'] = (base_lr + opt['lr']) / 2
                base_lr = opt['lr']

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
    model.define_optimizer(cfg)

    # define checkpoint
    checkpoint_dir = vars(cfg.model) if not cfg.model.use_smp else vars(cfg.model.smp)
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