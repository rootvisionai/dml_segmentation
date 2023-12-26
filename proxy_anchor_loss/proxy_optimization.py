import torch
import tqdm
import os


class ProxyOptimization(torch.nn.Module):

    def __init__(self, lr=0.1, max_steps=100, device="cuda"):
        super().__init__()

        self.device = device
        self.lr = lr
        self.max_steps = max_steps
        self.candidate_proxies_dict = {}

    def generate_proxy_dictionary(self, model, dataloader, overwrite=False):
        if list(self.candidate_proxies_dict.keys()) != dataloader.dataset.classes or overwrite:
            model.eval()
            candidate_proxies_dict = {}
            pbar = tqdm.tqdm(enumerate(dataloader))
            for batch_id, batch in pbar:
                with torch.no_grad():
                    temp = model(batch["image"].to(self.device)).to("cpu")
                    temp = self.l2_norm(temp)
                for sample_id in range(temp.shape[0]):
                    if batch["label_str"][sample_id] not in candidate_proxies_dict:
                        candidate_proxies_dict[batch["label_str"][sample_id]] = []
                    candidate_proxies_dict[batch["label_str"][sample_id]].append(temp[sample_id])
                pbar.set_description(f"CANDIDATE PROXY GENERATION: [{batch_id}/{len(dataloader)}]")

            proxies = []
            for key in candidate_proxies_dict:
                candidate_proxies_dict[key] = torch.stack(candidate_proxies_dict[key], dim=0)
                proxies.append(candidate_proxies_dict[key].sum(dim=0))

            self.candidate_proxies_dict = candidate_proxies_dict
            self.proxies = torch.nn.Parameter(torch.stack(proxies, dim=0).detach().to(self.device))

            model.train()

    def define_optimizer(self):
        self.optimizer = torch.optim.AdamW(params=[self.proxies], lr=self.lr)

    @staticmethod
    def criterion(similarity):
        # makes the angles between proxies maximum.
        loss = -torch.log(1 - similarity) * 2
        loss = torch.clamp(loss, min=0, max=8)
        return loss.sum()

    def sim_func(self, vectors):
        sim_mat = torch.nn.functional.linear(vectors.to(self.device), vectors.to(self.device))
        similarity_vector = torch.triu(sim_mat,
                                       diagonal=1)
        combinations = torch.nonzero(similarity_vector, as_tuple=True)
        similarity_vector = similarity_vector[combinations]
        return similarity_vector

    def optimize_full(self):
        pbar = tqdm.tqdm(range(self.max_steps))
        for i in pbar:
            loss, angle = self.optimize_step(i)
            text = f"ITER [{i}] | LOSS [{loss}] | MIN ANGLE [{angle}]"
            pbar.set_description(text)
        return True

    def optimize_step(self, i):
        self.optimizer.zero_grad()
        distance_vector = self.sim_func(self.l2_norm(self.proxies))

        if distance_vector.mean() > 0.9 and i == 0:
            distance_vector -= 0.1

        loss = self.criterion(distance_vector)
        loss.backward()
        self.optimizer.step()

        angle = torch.rad2deg(torch.acos(torch.clip(distance_vector, -0.9999, 0.9999))).detach().cpu().numpy()
        return loss.item(), min(angle)

    @staticmethod
    def l2_norm(vector):
        v_norm = vector.norm(dim=-1, p=2)
        vector = vector.divide(v_norm.unsqueeze(1))
        return vector

    def load_checkpoint(self, checkpoint_dir):
        if os.path.isfile(os.path.join(checkpoint_dir, "proxies.pth")):
            ckpt_dict = torch.load(os.path.join(checkpoint_dir, "proxies.pth"), map_location=self.device)
            self.proxies = torch.nn.Parameter(ckpt_dict["proxies"]).to(self.device)
            self.candidate_proxies_dict = ckpt_dict["candidate_proxies_dict"]
            return True
        else:
            return False

    def save_checkpoint(self, checkpoint_dir):
        torch.save({
            "proxies": self.proxies.detach(),
            "candidate_proxies_dict": self.candidate_proxies_dict,
        }, os.path.join(checkpoint_dir, "proxies.pth"))
        return True

if __name__=="__main__":
    POP = ProxyOptimization(lr=0.1, max_steps=100, device="cuda")
    POP.candidate_proxies_dict = {
        "class_0": torch.rand(4, 512),
        "class_1": torch.rand(4, 512)
    }
    proxies = []
    for key in POP.candidate_proxies_dict:
        proxies.append(POP.candidate_proxies_dict[key].sum(dim=0))
    POP.proxies = torch.nn.Parameter(POP.l2_norm(torch.stack(proxies, dim=0)))
    POP.proxies.requires_grad = True
    POP.define_optimizer()
    POP.optimize_full()

