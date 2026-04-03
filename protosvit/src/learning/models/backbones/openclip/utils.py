import timm
import torch
import torch.nn.functional as F

model_registry = {
    "open_clip_large": "vit_large_patch14_clip_224.laion2b",
    "open_clip_huge": "vit_huge_patch14_clip_224.laion2b",
    "open_clip_giant": "vit_giant_patch14_clip_224.laion2b",
    "open_clip_gigantic": "vit_gigantic_patch14_clip_224.laion2b",
}


def create_model(arch: str):
    model = timm.create_model(
        model_registry[arch], pretrained=True, num_classes=0, global_pool=""
    )
    return model
