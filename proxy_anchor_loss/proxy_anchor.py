import torch
import tqdm
import torch.nn.functional as F
import sklearn.preprocessing


def binarize(T, nb_classes):
    device = T.device
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).to(device)
    return T


def l2_norm(vector):
    v_norm = vector.norm(dim=-1, p=2).detach()
    vector = vector.divide(v_norm.unsqueeze(1))
    return vector


class ProxyAnchorLoss(torch.nn.Module):
    def __init__(self, alpha, mrg, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes
        self.alpha = alpha
        self.mrg = mrg

    def forward(self, X, T, proxies=None):
        # proxies = proxies.detach()
        cos = F.linear(l2_norm(X), l2_norm(proxies)).to(torch.float32)
        # cos_np = cos.detach().cpu().numpy()

        P_one_hot = T  # binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos - (1 - self.mrg)))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)

        P_one_hot_np = P_one_hot.detach().cpu().numpy()
        # P_one_hot[:, 0] = 0
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        if torch.isnan(loss):
            pass

        return loss