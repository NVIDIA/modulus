from typing import Union

from modulus import Module
from modulus.models.afno import AFNO


default_model_params = {
    "afno": {
        "patch_size": (8, 8),
        "embed_dim": 768,
        "depth": 12,
        "num_blocks": 8,
    }
}

def setup_model(
    model_type: str = "afno",
    model_name: Union[str,None] = None,
    **model_cfg
) -> Module:
    """Setup model from config dict."""
    model_kwargs = default_model_params[model_type].copy()
    model_kwargs.update(model_cfg)

    if model_type == "afno":
        model = AFNO(
            **model_kwargs
        )
    # TODO: add other model types

    if model_name is not None:
        model.meta.name = model_name

    return model
