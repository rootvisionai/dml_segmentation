import logging
import torch.nn as nn
import numpy as np
import torch
from .lion import Lion
from proxy_anchor_loss import ProxyOptimization, ProxyAnchorLoss

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'

    def load_checkpoint(self, path, device="cuda", load_opt=True):
        ckpt_dict = torch.load(path, map_location=device)
        self.load_state_dict(ckpt_dict["model_state_dict"]) if "model_state_dict" in ckpt_dict else 0
        if load_opt:
            self.optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"]) if "optimizer_state_dict" in ckpt_dict else 0
        print("loaded checkpoint:", path)
        return ckpt_dict["last_epoch"]

    def save_checkpoint(self, path, epoch):
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_epoch": epoch
        }, path)
        return True

    def training_step(self, x, m, proxies=None):
        x = self.forward(x)

        if isinstance(proxies, torch.Tensor):
            x = self.flatten(x)
            m = self.flatten(m)
            loss = self.calculate_loss(x, m, proxies)
        else:
            loss = self.calculate_loss(x, m)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def infer(self, x):
        with torch.no_grad():
            x = self.forward(x)
            x = torch.nn.functional.softmax(x, dim=1)
            x_ = x.argmax(1).unsqueeze(1)
        return x_.to(torch.float32), x

    def infer_sigmoid(self, x):
        with torch.no_grad():
            x = self.forward(x)
            x = torch.sigmoid(x)
        return x

    def define_loss_function(self, loss_function):
        self.calculate_loss = loss_function

    def define_optimizer(self, cfg):
        self.optimizer = getattr(torch.optim, cfg.training.optimizer)(
            params=[{"params": self.parameters()}],
            lr=cfg.training.learning_rate
        ) if cfg.training.optimizer != "Lion" else Lion(
            params=[{"params": self.parameters()}],
            lr=cfg.training.learning_rate
        )

    def flatten(self, x):
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def generate_proxies(self, dl_ev, device="cpu", proxy_anchor_cfg=None):
        candidates = {}
        for i, (image, mask) in enumerate(dl_ev):
            with torch.no_grad():
                emb = self.forward(image.to(device))

                if i == 0:
                    num_of_classes = mask.shape[1]
                    emb_size = emb.shape[1]

                emb = self.flatten(emb)
                mask = self.flatten(mask).argmax(dim=1)

                unqs = torch.unique(mask)
                for unq in unqs:
                    cand = emb[torch.where(mask == unq)].cpu()
                    if not unq.item() in candidates:
                        candidates[unq.item()] = cand
                    else:
                        candidates[unq.item()] = torch.cat([candidates[unq.item()], cand], dim=0)

            print(f"Generating candidate [{i}/{len(dl_ev)}] for proxies...")

        POP = ProxyOptimization(lr=proxy_anchor_cfg.lr, max_steps=proxy_anchor_cfg.steps, device="cuda")
        POP.candidate_proxies_dict = candidates

        random_proxies = torch.rand((num_of_classes, emb_size))
        proxies = []
        for key in range(num_of_classes):
            if not key in POP.candidate_proxies_dict:
                POP.candidate_proxies_dict[key] = random_proxies[key].unsqueeze(0)
            proxies.append(POP.candidate_proxies_dict[key].sum(dim=0))

        POP.proxies = torch.nn.Parameter(POP.l2_norm(torch.stack(proxies, dim=0)))
        POP.proxies.requires_grad = True
        POP.define_optimizer()
        POP.optimize_full()

        proxies = POP.proxies.to(device)
        return proxies
