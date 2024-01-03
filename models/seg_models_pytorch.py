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
            pooling='avg',
            dropout=0.2,
            activation='relu',
            **_
    ):
        super(SMP, self).__init__(out_layer=num_classes)

        aux_params = dict(
            pooling=pooling,  # one of 'avg', 'max'
            dropout=dropout,  # dropout ratio, default is None
            activation=activation,  # activation function, default is None
            classes=num_classes,  # define number of output labels
        )
        self.model = getattr(smp, model_name)(
            backbone=backbone,
            in_channels=in_channels,
            aux_params=aux_params,
            pretrained=pretrained
        )

    def forward(self, x):
        mask, _ = self.model(x)
        return mask