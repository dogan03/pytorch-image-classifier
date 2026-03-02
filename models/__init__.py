from models.base import BaseModel
from models.resnet import SimpleResNet, PretrainedResNet
from models.vgg import VGG

MODEL_REGISTRY = {
    "simple_resnet": SimpleResNet,
    "pretrained_resnet": PretrainedResNet,
    "vgg": VGG,
}


def get_model(name: str, **kwargs) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
