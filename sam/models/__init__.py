from . import base
from .lion import Lion


def load_model(cfg, pretrained=True):

    model = getattr(base, cfg.embeddings_model.backbone)(
        arch=cfg.embeddings_model.arch,
        embedding_size=cfg.embeddings_model.embedding_size,
        pretrained=pretrained,
        freeze_backbone=cfg.embeddings_model.freeze_backbone,
        emb_size_reduction_status=cfg.embeddings_model.emb_size_reduction_status,
        emb_size_reduction_heads=cfg.embeddings_model.emb_size_reduction_heads
    )

    return model
