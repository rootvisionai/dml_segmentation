import logging
import random

import torch.nn as nn
import numpy as np
import torch
import torchvision.utils

from .lion import Lion
from proxy_anchor_loss import ProxyOptimization
from kmeans_pytorch import kmeans


class BaseModel(nn.Module):
    def __init__(self, out_layer=256):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_final_conv = nn.Identity()
        self.base_final_conv = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(out_layer),
            nn.Conv2d(out_layer, out_layer, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.BatchNorm2d(out_layer),
            nn.Conv2d(out_layer, out_layer, kernel_size=1, padding=0, padding_mode="reflect")
        )

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
            self.optimizer.load_state_dict(
                ckpt_dict["optimizer_state_dict"]) if "optimizer_state_dict" in ckpt_dict else 0
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
        x = self.base_final_conv(x)

        # torchvision.utils.save_image(tensor=m.argmax(dim=1).unsqueeze(1)/81, fp="./debug_mask.png")
        # torchvision.utils.save_image(tensor=m[:, 0, :, :].unsqueeze(1) / m.max(), fp="./debug_mask.png")

        if isinstance(proxies, torch.Tensor):
            x = self.flatten(x)
            m = self.flatten(m)
            loss = self.calculate_loss(x, m, proxies)
        else:
            loss = self.calculate_loss(x, m)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def forward_all(self, x):
        x = self.forward(x)
        x = self.base_final_conv(x)
        return x

    def infer_sigmoid(self, x):
        with torch.no_grad():
            x = self.forward(x)
            x = self.base_final_conv(x)
            x = torch.sigmoid(x)
        return x

    def infer_kmeans(self, x, num_clusters):
        with torch.no_grad():
            x = self.forward(x)
            x = self.base_final_conv(x)

            preds = []
            for bs in range(x.shape[0]):
                x_ = x[bs].unsqueeze(0)
                original_shape = x_.shape
                x_ = self.flatten(x_)
                cluster_ids_x, cluster_centers = kmeans(X=x_.cuda(), num_clusters=num_clusters, distance='cosine',
                                                        device=torch.device('cuda'))
                pred = cluster_ids_x.reshape((original_shape[0], original_shape[2], original_shape[3]))
                pred = torch.stack([((pred == i) * 1.) for i in torch.unique(pred)], dim=1).to(torch.float)
                preds.append(pred)

        return torch.stack(preds, dim=0)

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

    def predict_few_shot(self, si, sm, qi, k=1, threshold=0.9, device="cpu"):

        with torch.no_grad():
            bs, _, row, col = qi.shape

            s_emb = torch.nn.functional.normalize(
                self.flatten(
                    self.base_final_conv(self.forward(si.to(torch.float32).to(device)))
                ),
                2, 1
            )

            q_emb = torch.nn.functional.normalize(
                self.flatten(
                    self.base_final_conv(self.forward(qi.to(torch.float32).to(device)))
                ),
                2, 1
            )

            s_emb = s_emb[torch.arange(0, s_emb.shape[0], si.shape[0])]
            cos_sim = torch.nn.functional.linear(q_emb, s_emb)

            sm = self.flatten(sm.to(torch.int8).to(device)).argmax(dim=1)

            topk = cos_sim.topk(1 + k)
            probs = topk[0][:, 1:].mean(dim=1)
            Y = sm[topk[1][:, 1:]]
            Y = Y.view(-1, Y.shape[-1])

            output = torch.unique(Y, sorted=True, dim=1)
            output = output[:, -1]

            ind = torch.where(probs < threshold)
            output[ind] = torch.tensor(0).to(device)

            output = output.reshape((bs, 1, row, col))

        return output

    def generate_temp_proxy(self, image, mask, proxy_cfg, device="cpu"):
        candidates = {}
        with torch.no_grad():
            mask_ = self.flatten(mask).argmax(dim=1)
            unqs = torch.unique(mask_)

            emb = self.forward(image)
            emb = self.base_final_conv(emb)

            emb = nn.functional.normalize(self.flatten(emb), 2, 1)

            for unq in unqs:
                if unq == 0:
                    continue
                cand = emb[torch.where(mask_ == unq)].cpu()
                if not unq.item() in candidates:
                    candidates[unq.item()] = cand
                else:
                    candidates[unq.item()] = torch.cat([candidates[unq.item()], cand])

        unq_ids = []
        for unq_id in candidates:
            candidates[unq_id] = candidates[unq_id].sum(dim=0)
            unq_ids.append(unq_id)

        POP = ProxyOptimization(lr=proxy_cfg.lr, max_steps=proxy_cfg.steps, device=device)
        POP.candidate_proxies_dict = candidates

        proxies = []
        for key in POP.candidate_proxies_dict:
            proxies.append(POP.candidate_proxies_dict[key])

        POP.proxies = torch.nn.Parameter(POP.l2_norm(torch.stack(proxies, dim=0).float()))
        POP.proxies.requires_grad = True
        POP.define_optimizer()
        POP.optimize_full()

        proxies = nn.functional.normalize(POP.proxies.to(device), 2, 1)
        if torch.isnan(proxies.sum()):
            pass
        return proxies.detach().to(torch.float16), unq_ids

    def generate_proxies(self, dl_ev, device="cpu", proxy_cfg=None):

        candidates = {}
        for i, (image, mask) in enumerate(dl_ev):
            with torch.no_grad():
                image = image.to(torch.float32).to(device)
                mask = mask.to(torch.int8).to(device)

                mask_ = self.flatten(mask).argmax(dim=1)
                unqs = torch.unique(mask_)

                # condition = any([not unq.item() in candidates for unq in unqs])
                condition = True
                if condition:
                    print(f"Generating candidate [{i}/{len(dl_ev)}] for proxies...")
                    emb = self.forward(image)
                    emb = self.base_final_conv(emb)

                    emb = nn.functional.normalize(self.flatten(emb), 2, 1)

                    for unq in unqs:
                        cand = emb[torch.where(mask_ == unq)].cpu()
                        cand = cand[torch.randint(low=0, high=cand.shape[0]-1, size=(int(cand.shape[0]*0.25), ))] if cand.shape[0] > 1000 else cand
                        if not unq.item() in candidates:
                            candidates[unq.item()] = cand
                        else:
                            candidates[unq.item()] = torch.cat([candidates[unq.item()], cand])
                        candidates[unq.item()] = candidates[unq.item()][torch.randint(low=0, high=cand.shape[0]-1, size=(5000, ))] if cand.shape[0] > 5000 else cand
                        print(f"{unq.item()} -> {candidates[unq.item()].shape[0]}")
                else:
                    print(f"[{i}/{len(dl_ev)}] | {len(candidates)}/{len(dl_ev.dataset.label_map)}")
                    if len(dl_ev.dataset.label_map) == len(candidates):
                        pass

        print("Candidates are generated.")

        unq_ids = []
        for unq_id in candidates:
            candidates[unq_id] = candidates[unq_id].sum(dim=0)
            unq_ids.append(unq_id)

        POP = ProxyOptimization(lr=proxy_cfg.lr, max_steps=proxy_cfg.steps, device=device)
        POP.candidate_proxies_dict = candidates

        proxies = []
        for key in POP.candidate_proxies_dict:
            proxies.append(POP.candidate_proxies_dict[key])

        POP.proxies = torch.nn.Parameter(POP.l2_norm(torch.stack(proxies, dim=0).float()))
        POP.proxies.requires_grad = True
        POP.define_optimizer()
        POP.optimize_full()

        proxies = nn.functional.normalize(POP.proxies.to(device), 2, 1)
        return proxies.detach()
