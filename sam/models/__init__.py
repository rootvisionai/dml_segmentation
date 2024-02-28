from . import base
from .lion import Lion


def load_model(cfg, pretrained=True):

    model = getattr(base, cfg.model.backbone)(
        arch=cfg.model.arch,
        embedding_size=cfg.model.embedding_size,
        pretrained=pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
        emb_size_reduction_status=cfg.model.emb_size_reduction_status,
        emb_size_reduction_heads=cfg.model.emb_size_reduction_heads
    )

    return model
