"""Default locations of the ProtoViT checkpoint, its prototype images and the CUB data.

Defaults follow the standard ProtoViT layout (paths relative to the ProtoViT/ directory). Each one can
be overridden with an environment variable, so the scripts run unchanged on any machine:

  PROTOVIT_CHECKPOINT      ProtoViT CUB checkpoint (.pth)
  PROTOVIT_PROTOTYPE_DIR   directory with prototype-img-original<p>.png and bb4.npy (push output)
  CUB_ROOT                 CUB_200_2011 (images.txt, bounding_boxes.txt, parts/part_locs.txt, images/)
  CUB_TEST_CROPPED         clean bounding-box-cropped CUB test images (<class>/<file>.jpg)
  CUB_SEGMENTATIONS        CUB segmentation masks (<class>/<file>.png)
"""
import os
from pathlib import Path

PV = Path(__file__).resolve().parent.parent          # the ProtoViT/ directory


def _path(env, default):
    return Path(os.environ[env]) if os.environ.get(env) else default


CHECKPOINT = _path("PROTOVIT_CHECKPOINT", PV / "saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth")
PROTOTYPE_DIR = _path("PROTOVIT_PROTOTYPE_DIR", PV / "saved_models/deit_small_patch16_224/exp1/img/epoch-4")
CUB_ROOT = _path("CUB_ROOT", PV / "datasets/CUB_200_2011")
CUB_TEST_CROPPED = _path("CUB_TEST_CROPPED", PV / "datasets/cub200_cropped/test_cropped")
CUB_SEGMENTATIONS = _path("CUB_SEGMENTATIONS", PV / "datasets/cub_segmentations_download/segmentations")
