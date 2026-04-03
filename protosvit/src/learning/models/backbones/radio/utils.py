import torch
from transformers import AutoModel

model_registry = {
    "radio": "radio_v2.1",
    "eradio": "e-radio_v2",
}


def create_model(arch: str, img_size=None):
    model = torch.hub.load(
        "NVlabs/RADIO",
        "radio_model",
        version=model_registry[arch],
        progress=True,
        skip_validation=True,
    )

    # conditioner = model.make_preprocessor_external()

    if "e-radio" in arch:
        model.model.set_optimal_window_size(
            (img_size, img_size)
        )  # where it expects a tuple of (height, width) of the input image.
    return model
