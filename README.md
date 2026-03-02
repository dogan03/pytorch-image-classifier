# PyTorch Image Classification Suite

A modular PyTorch image classification project supporting multiple model architectures (ResNet, VGG), configurable training pipelines, and a clean multi-file structure designed for real-world use.

## Project Structure

```
pytorch_image_classifier/
├── models/
│   ├── base.py          # Abstract base model
│   ├── resnet.py        # Custom & pretrained ResNet
│   ├── vgg.py           # VGG with configurable depth
│   └── __init__.py      # Model registry
├── data/
│   ├── dataset.py       # ImageFolderDataset
│   ├── transforms.py    # Train/val/test augmentation pipelines
│   ├── split.py         # Train/val/test splitting utility
│   └── __init__.py
├── utils/
│   ├── metrics.py       # AverageMeter, accuracy, confusion matrix
│   ├── checkpoint.py    # Save/load checkpoints
│   ├── logger.py        # Structured training logger
│   └── __init__.py
├── configs/
│   ├── resnet.yaml      # ResNet training config
│   └── vgg.yaml         # VGG training config
├── tests/
│   ├── test_models.py   # Model unit tests
│   └── test_data.py     # Data pipeline tests
├── train.py             # Training entry point
├── eval.py              # Evaluation on test set
├── predict.py           # Single/batch inference
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Prepare Data

Organize your dataset as:
```
data/
  train/class_a/  class_b/
  val/class_a/    class_b/
  test/class_a/   class_b/
```

Or use the built-in splitter:
```python
from data.split import split_dataset
split_dataset("raw_data/", "data/", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### Train

```bash
python train.py --config configs/resnet.yaml
python train.py --config configs/vgg.yaml --epochs 50 --lr 0.01
```

### Evaluate

```bash
python eval.py --config configs/resnet.yaml --checkpoint checkpoints/resnet/best_model.pth --confusion_matrix
```

### Predict

```bash
python predict.py --config configs/resnet.yaml --checkpoint checkpoints/resnet/best_model.pth --input image.jpg
python predict.py --config configs/resnet.yaml --checkpoint checkpoints/resnet/best_model.pth --input images/ --output results.json
```

### Run Tests

```bash
pytest tests/ -v
```

## Models

| Model            | Description                              |
|------------------|------------------------------------------|
| `simple_resnet`  | Lightweight custom ResNet (CIFAR-scale)  |
| `pretrained_resnet` | Torchvision ResNet18/34/50 with custom head |
| `vgg`            | VGG11/13/16/19 with optional batch norm  |

## Config Reference

```yaml
model:
  name: simple_resnet      # Model name from registry
  num_classes: 10

data:
  train_dir: data/train
  val_dir: data/val
  test_dir: data/test
  image_size: 32
  normalize: cifar         # 'cifar' or 'imagenet'
  num_workers: 4

training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 5.0e-4
  lr_scheduler: cosine     # 'cosine' or 'step'
```
