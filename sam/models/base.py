import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from collections import OrderedDict


def l2_norm(vector):
    v_norm = vector.norm(dim=-1, p=2)
    vector = vector.divide(v_norm.unsqueeze(1))
    return vector

class DoublePool(nn.Module):
    def __init__(self, out_size=1):
        super().__init__()
        self.out_size = out_size
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.avg_pooling(x)+self.max_pooling(x)
        return x

class EmbeddingSizeReduction(nn.Module):
    def __init__(self, pre_dim=2048, dim=512):
        super().__init__()
        self.pre_dim = pre_dim
        self.dim = dim

        self.to_attentions = nn.Sequential(nn.Linear(pre_dim, int(pre_dim / dim), bias=False),
                                           nn.Softmax(dim=-1))

    def forward(self, x):
        x = l2_norm(x)
        a = self.to_attentions(x)
        x = x.reshape(x.shape[0], int(self.pre_dim / self.dim), self.dim)
        x = x * a[..., None]
        return x.sum(1)


class Base(nn.Module):
    def get_params(self):
        if self.freeze_backbone:
            return [
                {"params": self.model.embedding.parameters()},
                {"params": self.model.emb_size_reduction.parameters()}
            ]
        else:
            return [
                {"params": self.model.parameters()}
            ]

    def train(self, train=True):
        if train:
            if self.freeze_backbone:
                self.model.eval()
                self.model.embedding.train()
                self.model.emb_size_reduction.train()
            else:
                self.model.train()
        else:
            self.model.train(False)

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.model(x)
        else:
            x = self.model(x)
        x = self.model.embedding(x)
        # x = self.model.emb_size_reduction(x)
        x = self.classifier(x)
        return x

    def training_step(self, image, label, proxies=None):
        embeddings = self.forward(image)
        loss = self.calculate_loss(embeddings, label, proxies)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def classifier_training_step(self, image, label):
        with torch.no_grad():
            if self.freeze_backbone:
                with torch.no_grad():
                    x = self.model(image)
            else:
                x = self.model(image)
            x = self.model.embedding(x)
            x = self.model.emb_size_reduction(x)
        x = self.classifier(x)
        loss = self.calculate_loss(x, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _initialize_weights(self, embedding):
        init.kaiming_normal_(embedding.weight, mode='fan_out')
        init.constant_(embedding.bias, 0)

    def load_checkpoint(self, path, device="cuda", load_optimizer=True):
        ckpt_dict = torch.load(path, map_location=device)
        self.load_state_dict(ckpt_dict["model_state_dict"]) if "model_state_dict" in ckpt_dict else 0
        try:
            if load_optimizer:
                self.optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"]) if "optimizer_state_dict" in ckpt_dict else 0
        except:
            pass
        print("loaded checkpoint:", path)
        return ckpt_dict["last_epoch"]

    def save_checkpoint(self, path, epoch):
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_epoch": epoch+1
        }, path)
        return True


class SwinTransformer(Base):
    def __init__(
            self,
            arch,
            embedding_size,
            pretrained=True,
            freeze_backbone=True,
            emb_size_reduction_status=False,
            emb_size_reduction_heads=1
    ):
        super(SwinTransformer, self).__init__()

        arch_options = {
            "tiny" : models.swin_t(pretrained=pretrained),
            "small": models.swin_s(pretrained=pretrained),
            "base" : models.swin_b(pretrained=pretrained)
        }

        self.freeze_backbone = freeze_backbone
        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.head.in_features
        self.model.embedding = nn.Linear(
            self.num_ftrs,
            self.embedding_size * emb_size_reduction_heads if emb_size_reduction_status else self.embedding_size
        )
        self.model.head = torch.nn.Identity()
        self.classifier = nn.Identity()
        self.model.avgpool = DoublePool(1)

        if emb_size_reduction_status:
            self.model.emb_size_reduction = EmbeddingSizeReduction(
                pre_dim=embedding_size * emb_size_reduction_heads,
                dim=embedding_size
            )
        else:
            self.model.emb_size_reduction = nn.Identity()

        self._initialize_weights(self.model.embedding)


class ConvNext(Base):
    def __init__(
            self,
            arch,
            embedding_size,
            pretrained=True,
            freeze_backbone=True,
            emb_size_reduction_status=False,
            emb_size_reduction_heads=1
    ):
        super(ConvNext, self).__init__()

        arch_options = {
            "tiny" : models.convnext_tiny(pretrained=pretrained),
            "small": models.convnext_small(pretrained=pretrained),
            "base" : models.convnext_base(pretrained=pretrained),
            "large": models.convnext_large(pretrained=pretrained),
        }

        self.freeze_backbone = freeze_backbone
        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.num_ftrs = self.model.classifier[-1].in_features
        self.model.embedding = nn.Linear(
            self.num_ftrs,
            self.embedding_size * emb_size_reduction_heads if emb_size_reduction_status else self.embedding_size
        )
        self.embedding_size = embedding_size
        self.model.classifier[-1] = torch.nn.Identity()
        self.model.avgpool = DoublePool(1)

        if emb_size_reduction_status:
            self.model.emb_size_reduction = EmbeddingSizeReduction(
                pre_dim=embedding_size * emb_size_reduction_heads,
                dim=embedding_size
            )
        else:
            self.model.emb_size_reduction = nn.Identity()

        self.classifier = nn.Identity()

        self._initialize_weights(self.model.embedding)


class ResNet(Base):
    def __init__(
            self,
            arch,
            embedding_size,
            pretrained=True,
            freeze_backbone=True,
            emb_size_reduction_status=False,
            emb_size_reduction_heads=1
    ):
        super(ResNet, self).__init__()

        arch_options = {
            "resnet18" : models.resnet18(pretrained=pretrained),
            "resnet34" : models.resnet34(pretrained=pretrained),
            "resnet50" : models.resnet50(pretrained=pretrained),
            "resnet101": models.resnet101(pretrained=pretrained),
            "resnet152": models.resnet152(pretrained=pretrained),
        }

        self.freeze_backbone = freeze_backbone
        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.num_ftrs = self.model.fc.in_features
        self.model.embedding = nn.Linear(
            self.num_ftrs, self.embedding_size * emb_size_reduction_heads if emb_size_reduction_status else self.embedding_size
        )
        self.embedding_size = embedding_size
        self.model.fc = torch.nn.Identity()
        self.model.avgpool = DoublePool(1)

        if emb_size_reduction_status:
            self.model.emb_size_reduction = EmbeddingSizeReduction(
                pre_dim=embedding_size * emb_size_reduction_heads,
                dim=embedding_size
            )
        else:
            self.model.emb_size_reduction = nn.Identity()

        self.classifier = nn.Identity()

        self._initialize_weights(self.model.embedding)


class SqueezeNet(Base):
    def __init__(
            self,
            arch,
            embedding_size,
            pretrained=True,
            freeze_backbone=True,
            emb_size_reduction_status=False,
            emb_size_reduction_heads=1
    ):
        super(SqueezeNet, self).__init__()

        arch_options = {
            "squeezenet1_0": models.squeezenet1_0(pretrained=pretrained),
            "squeezenet1_1": models.squeezenet1_1(pretrained=pretrained)
        }

        self.freeze_backbone = freeze_backbone
        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.num_ftrs = self.model.classifier[-1].in_features
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * emb_size_reduction_heads \
            if emb_size_reduction_status else self.embedding_size)
        self.embedding_size = embedding_size
        self.model.classifier = torch.nn.Identity()

        if emb_size_reduction_status:
            self.model.emb_size_reduction = EmbeddingSizeReduction(
                pre_dim=embedding_size * emb_size_reduction_heads,
                dim=embedding_size
            )
        else:
            self.model.emb_size_reduction = nn.Identity()

        self.classifier = nn.Identity()

        self._initialize_weights(self.model.embedding)


class EffNetV2(Base):
    def __init__(
            self,
            arch,
            embedding_size,
            pretrained=True,
            freeze_backbone=True,
            emb_size_reduction_status=False,
            emb_size_reduction_heads=1
    ):
        super(EffNetV2, self).__init__()

        arch_options = {
            "efficientnet_v2_s": models.efficientnet_v2_s(pretrained=pretrained),
            "efficientnet_v2_m": models.efficientnet_v2_s(pretrained=pretrained),
            "efficientnet_v2_l": models.efficientnet_v2_s(pretrained=pretrained)
        }

        self.freeze_backbone = freeze_backbone
        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.num_ftrs = self.model.classifier[-1].in_features
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * emb_size_reduction_heads \
            if emb_size_reduction_status else self.embedding_size)
        self.embedding_size = embedding_size

        self.model.classifier = torch.nn.Identity()
        self.model.avgpool = DoublePool(1)

        if emb_size_reduction_status:
            self.model.attention = EmbeddingSizeReduction(
                pre_dim=embedding_size * emb_size_reduction_heads,
                dim=embedding_size
            )
        else:
            self.model.attention = nn.Identity()

        self.classifier = nn.Identity()

        self._initialize_weights(self.model.embedding)


class ViT(Base):
    def __init__(
            self,
            arch,
            embedding_size,
            pretrained=True,
            freeze_backbone=True,
            emb_size_reduction_status=False,
            emb_size_reduction_heads=1
    ):
        super(ViT, self).__init__()

        arch_options = {
            "vit_b_16": models.vit_b_16(pretrained=pretrained),
            "vit_b_32": models.vit_b_32(pretrained=pretrained),
            "vit_l_16": models.vit_l_16(pretrained=pretrained),
            "vit_l_32": models.vit_l_32(pretrained=pretrained)
            # "vit_h_14": models.vit_h_14(pretrained=pretrained)
        }

        self.freeze_backbone = freeze_backbone
        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.num_ftrs = self.model.heads.head.in_features
        self.model.embedding = nn.Linear(
            self.num_ftrs,
            self.embedding_size * emb_size_reduction_heads if emb_size_reduction_status else self.embedding_size
        )
        self.embedding_size = embedding_size
        self.model.heads = torch.nn.Identity()

        if emb_size_reduction_status:
            self.model.attention = EmbeddingSizeReduction(pre_dim=embedding_size * emb_size_reduction_heads, dim=embedding_size)
        else:
            self.model.attention = nn.Identity()

        self.classifier = nn.Identity()

        self._initialize_weights(self.model.embedding)