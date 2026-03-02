"""
predict.py - Run inference on a single image or a directory of images.

Usage:
    python predict.py --config configs/resnet.yaml --checkpoint checkpoints/resnet/best_model.pth --input image.jpg
    python predict.py --config configs/resnet.yaml --checkpoint checkpoints/resnet/best_model.pth --input images/
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image

from data.transforms import get_inference_transforms
from models import get_model
from utils import load_checkpoint, get_logger


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict_single(model, image_path: Path, transform, class_names: list, device) -> dict:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(min(5, len(class_names)))
    return {
        "file": str(image_path),
        "prediction": class_names[top_indices[0].item()],
        "confidence": round(top_probs[0].item() * 100, 2),
        "top5": [
            {"class": class_names[i.item()], "prob": round(p.item() * 100, 2)}
            for i, p in zip(top_indices, top_probs)
        ],
    }


def main(args):
    logger = get_logger("predict")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg["model"]
    model = get_model(
        model_cfg["name"],
        num_classes=model_cfg["num_classes"],
        **{k: v for k, v in model_cfg.items() if k not in ("name", "num_classes")},
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()

    data_cfg = cfg["data"]
    transform = get_inference_transforms(data_cfg["image_size"], data_cfg["normalize"])

    class_names = args.classes.split(",") if args.classes else [f"class_{i}" for i in range(model_cfg["num_classes"])]

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = [p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    results = []
    for path in sorted(image_paths):
        result = predict_single(model, path, transform, class_names, device)
        results.append(result)
        logger.info(f"{path.name:<40} => {result['prediction']} ({result['confidence']:.1f}%)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--classes", default=None, help="Comma-separated class names")
    parser.add_argument("--output", default=None, help="Save JSON results to this path")
    args = parser.parse_args()
    main(args)
