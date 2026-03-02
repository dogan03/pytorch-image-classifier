from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


def get_train_transforms(image_size: int = 224, normalize: str = "imagenet") -> transforms.Compose:
    mean, std = _get_norm_stats(normalize)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_val_transforms(image_size: int = 224, normalize: str = "imagenet") -> transforms.Compose:
    mean, std = _get_norm_stats(normalize)
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_test_transforms(image_size: int = 224, normalize: str = "imagenet") -> transforms.Compose:
    return get_val_transforms(image_size, normalize)


def get_inference_transforms(image_size: int = 224, normalize: str = "imagenet") -> transforms.Compose:
    return get_val_transforms(image_size, normalize)


def _get_norm_stats(preset: str):
    if preset == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    elif preset == "cifar":
        return CIFAR_MEAN, CIFAR_STD
    else:
        raise ValueError(f"Unknown normalization preset: {preset}. Use 'imagenet' or 'cifar'.")
