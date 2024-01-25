from .seg_models_pytorch import SMP

def load(cfg):
    model = SMP(
        num_classes=cfg.model.out_layer_size,
        in_channels=cfg.model.in_channels,
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained_weights,
        model_name=cfg.model.arch,
    )
    return model