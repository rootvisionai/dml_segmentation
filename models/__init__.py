from .seg_models_pytorch import SMP

def load(cfg):
    model = SMP(
        num_classes=cfg.model.smp.out_layer_size,
        in_channels=cfg.model.smp.in_channels,
        backbone=cfg.model.smp.backbone,
        pretrained=cfg.model.smp.pretrained_weights,
        model_name=cfg.model.smp.arch,
    )
    return model