from .base import HairSegmenter
from .bisenet import BiSeNetSegmenter

_MODELS: dict[str, type[HairSegmenter]] = {
    "bisenet": BiSeNetSegmenter,
}

_INSTANCES: dict[str, HairSegmenter] = {}


def get_segmenter(model_name: str) -> HairSegmenter:
    """Get a segmenter instance by name. Loads model on first call."""
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODELS.keys())}")

    if model_name not in _INSTANCES:
        instance = _MODELS[model_name]()
        instance.load()
        _INSTANCES[model_name] = instance

    return _INSTANCES[model_name]


def available_models() -> list[str]:
    return list(_MODELS.keys())
