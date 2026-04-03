import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained

PATH_MEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class ModifiedTimmModel(nn.Module):
    def __init__(self, timm_model):
        super(ModifiedTimmModel, self).__init__()
        self.timm_model = timm_model
        self.patch_size = timm_model.patch_embed.patch_size[0]
        self.embed_dim = timm_model.embed_dim

    def forward(self, x):
        x = self.timm_model.forward_features(x)
        return x


def create_model():
    model, preprocess = create_model_from_pretrained(PATH_MEDCLIP)
    vision_encoder = ModifiedTimmModel(model.visual.trunk)

    del model
    del preprocess
    return vision_encoder
