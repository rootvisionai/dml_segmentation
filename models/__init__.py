"""
model implementations are from: https://github.com/yassouali/pytorch-segmentation/tree/master/models
"""

from .unet import UNet, UNetResnet
from .deeplabv3_plus import DeepLab
from .deeplabv3_plus_xception import DeepLab as DeepLabX
from .seg_models_pytorch import SMP
import sys

def load(cfg):
    if cfg.use_smp:
        SMP(
            num_classes=cfg.model.smp.out_layer_size,
            in_channels=cfg.model.smp.in_channels,
            backbone=cfg.model.smp.backbone,
            pretrained=cfg.model.smp.pretrained_weights,
            model_name=cfg.model.smp.arch,
            pooling=cfg.model.smp.pooling,
            dropout=cfg.model.smp.dropout,
            activation=cfg.model.smp.activation
        )
    else:
        if cfg.model.arch == "DeepLabX":
            model = DeepLabX(
                num_classes=cfg.model.out_layer_size,
                in_channels=3,
                backbone=cfg.model.backbone,
                pretrained=True,
                output_stride=cfg.model.output_stride,
                freeze_bn=cfg.model.freeze_bn,
                freeze_backbone=cfg.model.freeze_backbone
            )
        elif cfg.model.arch == "DeepLab":
            model = DeepLab(
                num_classes=cfg.model.out_layer_size,
                in_channels=3,
                backbone=cfg.model.backbone,
                pretrained=True,
                output_stride=cfg.model.output_stride,
                freeze_bn=cfg.model.freeze_bn,
                freeze_backbone=cfg.model.freeze_backbone
            )
        elif cfg.model.arch == "UNetResnet":
            model = UNetResnet(
                num_classes=cfg.model.out_layer_size,
                in_channels=3,
                backbone=cfg.model.backbone,
                pretrained=True,
                freeze_bn=cfg.model.freeze_bn,
                freeze_backbone=cfg.model.freeze_backbone
            )
        elif cfg.model.arch == "UNet":
            model = UNet(
                num_classes=cfg.model.out_layer_size,
                in_channels=3,
                freeze_bn=cfg.model.freeze_bn
            )
        else:
            print(f"GIVEN {cfg.model.arch} is not implemented.")
            sys.exit(1)
        return model