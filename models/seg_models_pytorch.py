import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel
import segmentation_models_pytorch as smp

class SMP(BaseModel):
    def __init__(
            self,
            num_classes,
            in_channels=3,
            backbone='resnet50',
            pretrained=True,
            model_name='Unet',
            **_
    ):
        super(SMP, self).__init__(out_layer=num_classes)

        self.model = getattr(smp, "create_model")(
            arch=model_name,
            encoder_name=backbone,
            in_channels=in_channels,
            classes=num_classes,
            encoder_weights=pretrained
        )

    def forward(self, x):
        mask = self.model(x)
        return mask