"""Heterogeneous CNN ensemble for Fashion-MNIST.

Five architecturally different CNNs (wide SE-ResNet, Sobel edge-aware
SE-ResNet, multi-scale residual net, spatial VGG, tiny ConvNeXt) are combined
with learned class-wise weights and per-member temperatures.  A specialist
CNN re-scores the four most confused upper-body classes (T-shirt, Pullover,
Coat, Shirt) behind a calibrated gate, and test-time augmentation is applied
only to low-confidence inputs.

Training uses only the Fashion-MNIST training split; the test split is never
touched until final evaluation.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import struct
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Reproducibility and constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 10
IMAGE_SIZE = 28
FASHION_MEAN = 0.2860406
FASHION_STD = 0.3530242
UPPER_CLASSES = (0, 2, 4, 6)
CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "fashion_ensemble.pth"


def set_global_seed(seed: int) -> None:
    """Seed Python and PyTorch without forcing slow deterministic kernels."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# IDX data reader
# ---------------------------------------------------------------------------


def _read_idx_images(path: Path) -> Tensor:
    """Read an IDX image file as an uint8 tensor of shape N x 1 x 28 x 28."""

    with path.open("rb") as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16))

    if magic != 2051:
        raise ValueError(f"Invalid IDX image magic number in {path}: {magic}")
    if rows != IMAGE_SIZE or columns != IMAGE_SIZE:
        raise ValueError(
            f"Expected {IMAGE_SIZE} x {IMAGE_SIZE} images, got {rows} x {columns}."
        )

    expected_size = 16 + count * rows * columns
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Unexpected IDX image file size for {path}: "
            f"expected {expected_size}, got {actual_size}."
        )

    mapped = torch.from_file(
        str(path), shared=False, size=expected_size, dtype=torch.uint8
    )
    return mapped[16:].reshape(count, 1, rows, columns).clone()


def _read_idx_labels(path: Path) -> Tensor:
    """Read an IDX label file as an int64 tensor."""

    with path.open("rb") as file:
        magic, count = struct.unpack(">II", file.read(8))

    if magic != 2049:
        raise ValueError(f"Invalid IDX label magic number in {path}: {magic}")

    expected_size = 8 + count
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Unexpected IDX label file size for {path}: "
            f"expected {expected_size}, got {actual_size}."
        )

    mapped = torch.from_file(
        str(path), shared=False, size=expected_size, dtype=torch.uint8
    )
    return mapped[8:].to(torch.int64).clone()


class LocalFashionMNIST(Dataset):
    """Read a local Fashion-MNIST split stored as raw IDX files.

    Images are returned as float tensors in the raw [0, 1] range.  Normalizing
    and augmentation are performed in batches on the GPU during training.
    """

    def __init__(self, root: Path | str, split: str = "train", transform=None):
        self.root = Path(root)
        self.transform = transform

        prefixes = {"train": "train", "test": "t10k"}
        if split not in prefixes:
            raise ValueError("split must be 'train' or 'test'.")

        prefix = prefixes[split]
        image_path = self.root / f"{prefix}-images-idx3-ubyte"
        label_path = self.root / f"{prefix}-labels-idx1-ubyte"

        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(
                f"Missing Fashion-MNIST {split} files in {self.root}. "
                "Run `python scripts/download_data.py` first."
            )

        self.images = _read_idx_images(image_path)
        self.labels = _read_idx_labels(label_path)

        if len(self.images) != len(self.labels):
            raise ValueError("The number of images and labels does not match.")

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        image = self.images[index].to(torch.float32).div_(255.0)
        label = int(self.labels[index].item())

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class IndexedDataset(Dataset):
    """A view that additionally returns global indices and sample weights."""

    def __init__(
        self,
        base: LocalFashionMNIST,
        indices: Sequence[int] | Tensor,
        sample_weights: Optional[Tensor] = None,
    ):
        self.base = base
        self.indices = torch.as_tensor(indices, dtype=torch.long).clone()
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, position: int):
        global_index = int(self.indices[position].item())
        image, label = self.base[global_index]
        weight = 1.0
        if self.sample_weights is not None:
            weight = float(self.sample_weights[global_index].item())
        return image, label, global_index, weight


# ---------------------------------------------------------------------------
# Batch augmentation
# ---------------------------------------------------------------------------


class BatchAugmenter:
    """GPU-friendly mild geometric and photometric augmentation."""

    def __init__(
        self,
        degrees: float = 7.0,
        translate: float = 0.07,
        scale: float = 0.05,
        horizontal_flip: float = 0.5,
        brightness: float = 0.08,
        contrast: float = 0.10,
        erase_probability: float = 0.10,
    ):
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.horizontal_flip = float(horizontal_flip)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.erase_probability = float(erase_probability)

    def __call__(self, images: Tensor) -> Tensor:
        images = images.clone()
        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype

        if self.horizontal_flip > 0.0:
            flip_mask = torch.rand(batch_size, device=device) < self.horizontal_flip
            if bool(flip_mask.any()):
                images[flip_mask] = torch.flip(images[flip_mask], dims=(3,))

        if self.degrees > 0.0 or self.translate > 0.0 or self.scale > 0.0:
            angles = (
                torch.empty(batch_size, device=device, dtype=dtype)
                .uniform_(-self.degrees, self.degrees)
                .mul_(math.pi / 180.0)
            )
            scales = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
                1.0 - self.scale, 1.0 + self.scale
            )
            translations_x = torch.empty(
                batch_size, device=device, dtype=dtype
            ).uniform_(-2.0 * self.translate, 2.0 * self.translate)
            translations_y = torch.empty(
                batch_size, device=device, dtype=dtype
            ).uniform_(-2.0 * self.translate, 2.0 * self.translate)

            cosine = torch.cos(angles) * scales
            sine = torch.sin(angles) * scales
            theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
            theta[:, 0, 0] = cosine
            theta[:, 0, 1] = -sine
            theta[:, 1, 0] = sine
            theta[:, 1, 1] = cosine
            theta[:, 0, 2] = translations_x
            theta[:, 1, 2] = translations_y

            grid = F.affine_grid(theta, images.size(), align_corners=False)
            images = F.grid_sample(
                images,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

        if self.contrast > 0.0:
            factors = torch.empty(
                batch_size, 1, 1, 1, device=device, dtype=dtype
            ).uniform_(1.0 - self.contrast, 1.0 + self.contrast)
            means = images.mean(dim=(2, 3), keepdim=True)
            images = (images - means) * factors + means

        if self.brightness > 0.0:
            offsets = torch.empty(
                batch_size, 1, 1, 1, device=device, dtype=dtype
            ).uniform_(-self.brightness, self.brightness)
            images = images + offsets

        images.clamp_(0.0, 1.0)

        if self.erase_probability > 0.0:
            erase_mask = torch.rand(batch_size, device=device) < self.erase_probability
            erase_indices = torch.nonzero(erase_mask, as_tuple=False).flatten().tolist()
            for batch_index in erase_indices:
                height = int(torch.randint(2, 7, (1,), device=device).item())
                width = int(torch.randint(2, 7, (1,), device=device).item())
                top = int(
                    torch.randint(0, IMAGE_SIZE - height + 1, (1,), device=device).item()
                )
                left = int(
                    torch.randint(0, IMAGE_SIZE - width + 1, (1,), device=device).item()
                )
                images[batch_index, :, top : top + height, left : left + width] = 0.0

        return normalize_images(images)


def normalize_images(images: Tensor) -> Tensor:
    """Normalize raw [0, 1] Fashion-MNIST images."""

    return (images - FASHION_MEAN) / FASHION_STD


# ---------------------------------------------------------------------------
# CNN building blocks
# ---------------------------------------------------------------------------


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class StochasticDepth(nn.Module):
    """Drop a residual branch per sample during training."""

    def __init__(self, probability: float):
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return value
        keep_probability = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            shape, dtype=value.dtype, device=value.device
        )
        random_tensor.floor_()
        return value.div(keep_probability) * random_tensor


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, value: Tensor) -> Tensor:
        scale = self.pool(value)
        scale = F.silu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return value * scale


class PreActivationSEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.drop_path = StochasticDepth(drop_path)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, value: Tensor) -> Tensor:
        activated = F.silu(self.bn1(value), inplace=True)
        shortcut = value if self.shortcut is None else self.shortcut(activated)
        output = self.conv1(activated)
        output = self.conv2(F.silu(self.bn2(output), inplace=True))
        output = self.se(output)
        return shortcut + self.drop_path(output)


class WideSEResNet(nn.Module):
    """A high-resolution residual CNN that delays spatial downsampling."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        widths: Tuple[int, int, int] = (48, 96, 160),
        depths: Tuple[int, int, int] = (2, 2, 3),
        dropout: float = 0.20,
        max_drop_path: float = 0.08,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            conv3x3(in_channels, widths[0]),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(inplace=True),
        )

        total_blocks = sum(depths)
        block_number = 0
        stages: List[nn.Module] = []
        current_channels = widths[0]
        for stage_index, (width, depth) in enumerate(zip(widths, depths)):
            for inner_index in range(depth):
                stride = 2 if stage_index > 0 and inner_index == 0 else 1
                drop_path = max_drop_path * block_number / max(total_blocks - 1, 1)
                stages.append(
                    PreActivationSEBlock(
                        current_channels,
                        width,
                        stride=stride,
                        drop_path=drop_path,
                    )
                )
                current_channels = width
                block_number += 1
        self.features = nn.Sequential(*stages)
        self.final_bn = nn.BatchNorm2d(current_channels)
        self.classifier = nn.Sequential(
            nn.Linear(current_channels * 2, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, value: Tensor) -> Tensor:
        value = self.features(self.stem(value))
        value = F.silu(self.final_bn(value), inplace=True)
        average = F.adaptive_avg_pool2d(value, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(value, 1).flatten(1)
        return torch.cat((average, maximum), dim=1)

    def forward(self, value: Tensor) -> Tensor:
        return self.classifier(self.forward_features(value))


class SobelAdapter(nn.Module):
    """Create original, horizontal-edge, and vertical-edge channels."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).reshape(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3).contiguous()
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, value: Tensor) -> Tensor:
        horizontal = F.conv2d(value, self.sobel_x, padding=1)
        vertical = F.conv2d(value, self.sobel_y, padding=1)
        horizontal = torch.tanh(horizontal * 0.5)
        vertical = torch.tanh(vertical * 0.5)
        return torch.cat((value, horizontal, vertical), dim=1)


class EdgeAwareSEResNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.adapter = SobelAdapter()
        self.backbone = WideSEResNet(
            in_channels=3,
            num_classes=num_classes,
            widths=(40, 80, 144),
            depths=(2, 2, 3),
            dropout=0.18,
            max_drop_path=0.07,
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.backbone(self.adapter(value))


class MultiScaleBlock(nn.Module):
    """Residual block combining local and wider receptive fields."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        branch = out_channels // 3
        remainder = out_channels - branch * 2
        self.pre = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.branch3 = conv3x3(in_channels, branch, stride=stride)
        self.branch5 = nn.Conv2d(
            in_channels,
            branch,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
        )
        self.branch_dilated = conv3x3(
            in_channels, remainder, stride=stride, dilation=2
        )
        self.merge = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            SqueezeExcitation(out_channels),
        )
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, value: Tensor) -> Tensor:
        activated = self.pre(value)
        output = torch.cat(
            (
                self.branch3(activated),
                self.branch5(activated),
                self.branch_dilated(activated),
            ),
            dim=1,
        )
        shortcut = value if self.shortcut is None else self.shortcut(activated)
        return shortcut + self.merge(output)


class MultiScaleNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            conv3x3(1, 48), nn.BatchNorm2d(48), nn.SiLU(inplace=True)
        )
        self.features = nn.Sequential(
            MultiScaleBlock(48, 48),
            MultiScaleBlock(48, 48),
            MultiScaleBlock(48, 96, stride=2),
            MultiScaleBlock(96, 96),
            MultiScaleBlock(96, 160, stride=2),
            MultiScaleBlock(160, 160),
        )
        self.head = nn.Sequential(
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(160, num_classes),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.head(self.features(self.stem(value)))


class SpatialVGG(nn.Module):
    """VGG-like CNN retaining a 4 x 4 spatial descriptor before the head."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def block(in_channels: int, out_channels: int, count: int) -> nn.Sequential:
            layers: List[nn.Module] = []
            current = in_channels
            for _ in range(count):
                layers.extend(
                    (
                        conv3x3(current, out_channels),
                        nn.BatchNorm2d(out_channels),
                        nn.GELU(),
                    )
                )
                current = out_channels
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, 48, 2),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            block(48, 96, 2),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            block(96, 160, 3),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160 * 4 * 4, 384),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(384, num_classes),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.classifier(self.features(value))


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.epsilon = epsilon

    def forward(self, value: Tensor) -> Tensor:
        value = value.permute(0, 2, 3, 1)
        value = F.layer_norm(
            value, (value.shape[-1],), self.weight, self.bias, self.epsilon
        )
        return value.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, drop_path: float = 0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = LayerNorm2d(channels)
        self.pointwise1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.pointwise2 = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 1.0e-6)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, value: Tensor) -> Tensor:
        output = self.depthwise(value)
        output = self.norm(output)
        output = self.pointwise1(output)
        output = F.gelu(output)
        output = self.pointwise2(output)
        return value + self.drop_path(self.layer_scale * output)


class TinyConvNeXt(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Conv2d(1, 48, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(48, 0.00),
            ConvNeXtBlock(48, 0.02),
        )
        self.down1 = nn.Sequential(LayerNorm2d(48), nn.Conv2d(48, 96, 2, 2))
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(96, 0.03),
            ConvNeXtBlock(96, 0.05),
        )
        self.down2 = nn.Sequential(LayerNorm2d(96), nn.Conv2d(96, 160, 2, 2))
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(160, 0.06),
            ConvNeXtBlock(160, 0.08),
            ConvNeXtBlock(160, 0.10),
        )
        self.norm = nn.LayerNorm(160)
        self.head = nn.Linear(160, num_classes)

    def forward(self, value: Tensor) -> Tensor:
        value = self.stage1(self.stem(value))
        value = self.stage2(self.down1(value))
        value = self.stage3(self.down2(value))
        value = value.mean(dim=(2, 3))
        return self.head(self.norm(value))


def create_generalists() -> nn.ModuleList:
    """Create five architectures with intentionally different inductive biases."""

    return nn.ModuleList(
        [
            WideSEResNet(
                widths=(48, 96, 160),
                depths=(2, 2, 3),
                dropout=0.20,
                max_drop_path=0.08,
            ),
            EdgeAwareSEResNet(),
            MultiScaleNet(),
            SpatialVGG(),
            TinyConvNeXt(),
        ]
    )


class FashionEnsemble(nn.Module):
    """Five generalists plus a 0/2/4/6 upper-body specialist."""

    def __init__(self):
        super().__init__()
        self.members = create_generalists()
        self.specialist = WideSEResNet(
            num_classes=4,
            widths=(48, 96, 160),
            depths=(2, 2, 3),
            dropout=0.18,
            max_drop_path=0.06,
        )
        member_count = len(self.members)
        self.register_buffer(
            "class_weights", torch.full((member_count, NUM_CLASSES), 1.0 / member_count)
        )
        self.register_buffer("temperatures", torch.ones(member_count))
        self.register_buffer("class_bias", torch.zeros(NUM_CLASSES))
        self.register_buffer(
            "upper_indices", torch.tensor(UPPER_CLASSES, dtype=torch.long)
        )
        self.register_buffer("specialist_threshold", torch.tensor(0.55))
        self.register_buffer("specialist_margin", torch.tensor(0.35))
        self.register_buffer("specialist_alpha", torch.tensor(0.65))
        self.register_buffer("tta_confidence", torch.tensor(0.93))
        self.register_buffer("tta_margin", torch.tensor(0.35))

    def member_logits(self, images: Tensor) -> Tensor:
        return torch.stack([member(images) for member in self.members], dim=1)

    def combine_member_logits(self, logits: Tensor) -> Tensor:
        temperatures = self.temperatures.clamp(0.35, 4.0).reshape(1, -1, 1)
        calibrated = logits / temperatures
        weights = self.class_weights.reshape(1, *self.class_weights.shape)
        return (calibrated * weights).sum(dim=1) + self.class_bias

    def _apply_specialist(self, images: Tensor, probabilities: Tensor) -> Tensor:
        upper = self.upper_indices
        upper_probabilities = probabilities.index_select(1, upper)
        upper_mass = upper_probabilities.sum(dim=1)
        top_two = probabilities.topk(k=2, dim=1).values
        margin = top_two[:, 0] - top_two[:, 1]
        top_labels = probabilities.argmax(dim=1)
        top_is_upper = (top_labels.unsqueeze(1) == upper.unsqueeze(0)).any(dim=1)
        trigger = (
            (upper_mass >= self.specialist_threshold)
            & (margin <= self.specialist_margin)
            & top_is_upper
        )
        if not bool(trigger.any()):
            return probabilities

        specialist_probabilities = self.specialist(images[trigger]).softmax(dim=1)
        general_conditional = upper_probabilities[trigger] / upper_mass[trigger].unsqueeze(1).clamp_min(
            1.0e-8
        )
        alpha = self.specialist_alpha.clamp(0.0, 1.0)
        blended = (1.0 - alpha) * general_conditional + alpha * specialist_probabilities

        output = probabilities.clone()
        rows = torch.nonzero(trigger, as_tuple=False).flatten()
        output_upper = blended * upper_mass[trigger].unsqueeze(1)
        output[rows.unsqueeze(1), upper.unsqueeze(0)] = output_upper
        return output

    def predict_proba_once(self, images: Tensor) -> Tensor:
        logits = self.combine_member_logits(self.member_logits(images))
        probabilities = logits.softmax(dim=1)
        return self._apply_specialist(images, probabilities)

    @staticmethod
    def _shift(images: Tensor, horizontal: int = 0, vertical: int = 0) -> Tensor:
        shifted = torch.zeros_like(images)
        source_top = max(-vertical, 0)
        source_bottom = IMAGE_SIZE - max(vertical, 0)
        source_left = max(-horizontal, 0)
        source_right = IMAGE_SIZE - max(horizontal, 0)
        target_top = max(vertical, 0)
        target_bottom = IMAGE_SIZE - max(-vertical, 0)
        target_left = max(horizontal, 0)
        target_right = IMAGE_SIZE - max(-horizontal, 0)
        shifted[:, :, target_top:target_bottom, target_left:target_right] = images[
            :, :, source_top:source_bottom, source_left:source_right
        ]
        return shifted

    def predict_proba(self, images: Tensor, use_tta: bool = True) -> Tensor:
        original = self.predict_proba_once(images)
        if not use_tta:
            return original

        top_two = original.topk(k=2, dim=1).values
        confidence = top_two[:, 0]
        margin = top_two[:, 0] - top_two[:, 1]
        uncertain = (confidence < self.tta_confidence) | (margin < self.tta_margin)
        if not bool(uncertain.any()):
            return original

        selected = images[uncertain]
        transformed = [
            torch.flip(selected, dims=(3,)),
            self._shift(selected, horizontal=-1),
            self._shift(selected, horizontal=1),
            self._shift(selected, vertical=-1),
            self._shift(selected, vertical=1),
        ]
        averaged = self.predict_proba_once(selected)
        for variant in transformed:
            averaged = averaged + self.predict_proba_once(variant)
        averaged = averaged / float(len(transformed) + 1)

        output = original.clone()
        output[uncertain] = averaged
        return output

    def forward(self, images: Tensor) -> Tensor:
        return self.predict_proba_once(images).clamp_min(1.0e-8).log()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = float(decay)
        self.shadow: Dict[str, Tensor] = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for name, value in current.items():
            if not value.is_floating_point():
                self.shadow[name].copy_(value)
            else:
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


class WarmupCosineScheduler:
    """Per-step linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        minimum_ratio: float = 0.03,
    ):
        self.optimizer = optimizer
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 1)
        self.minimum_ratio = float(minimum_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_number = 0

    def step(self) -> None:
        self.step_number += 1
        if self.step_number <= self.warmup_steps:
            ratio = self.step_number / self.warmup_steps
        else:
            progress = (self.step_number - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            ratio = self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * ratio


def initialize_module(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm)):
            if layer.weight is not None:
                nn.init.ones_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def stratified_three_way_split(
    labels: Tensor,
    selection_ratio: float = 0.05,
    calibration_ratio: float = 0.05,
    seed: int = 2100,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return class-balanced train, model-selection, and calibration indices."""

    generator = torch.Generator().manual_seed(seed)
    train_parts: List[Tensor] = []
    selection_parts: List[Tensor] = []
    calibration_parts: List[Tensor] = []

    for class_index in range(NUM_CLASSES):
        class_indices = torch.nonzero(labels == class_index, as_tuple=False).flatten()
        permutation = torch.randperm(class_indices.numel(), generator=generator)
        class_indices = class_indices[permutation]
        calibration_count = int(round(class_indices.numel() * calibration_ratio))
        selection_count = int(round(class_indices.numel() * selection_ratio))
        calibration_parts.append(class_indices[:calibration_count])
        selection_parts.append(
            class_indices[calibration_count : calibration_count + selection_count]
        )
        train_parts.append(class_indices[calibration_count + selection_count :])

    def shuffled(parts: List[Tensor]) -> Tensor:
        combined = torch.cat(parts)
        order = torch.randperm(combined.numel(), generator=generator)
        return combined[order]

    return shuffled(train_parts), shuffled(selection_parts), shuffled(calibration_parts)


def bayesian_bootstrap_weights(labels: Tensor, seed: int) -> Tensor:
    """Generate class-normalized exponential Bayesian-bootstrap weights."""

    generator = torch.Generator().manual_seed(seed)
    uniform = torch.rand(labels.numel(), generator=generator).clamp_min_(1.0e-6)
    weights = -torch.log(uniform)
    for class_index in range(NUM_CLASSES):
        mask = labels == class_index
        weights[mask] /= weights[mask].mean().clamp_min(1.0e-6)
    return weights.clamp_(0.25, 4.0)


def map_upper_labels(labels: Tensor) -> Tensor:
    mapped = torch.full_like(labels, -1)
    for new_label, original_label in enumerate(UPPER_CLASSES):
        mapped[labels == original_label] = new_label
    return mapped


def weighted_classification_loss(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor,
    label_smoothing: float = 0.02,
    mixup: Optional[Tuple[Tensor, Tensor, float]] = None,
) -> Tensor:
    if mixup is None:
        losses = F.cross_entropy(
            logits, labels, reduction="none", label_smoothing=label_smoothing
        )
        return (losses * weights).sum() / weights.sum().clamp_min(1.0e-8)

    second_labels, second_weights, coefficient = mixup
    first_loss = F.cross_entropy(
        logits, labels, reduction="none", label_smoothing=label_smoothing
    )
    second_loss = F.cross_entropy(
        logits, second_labels, reduction="none", label_smoothing=label_smoothing
    )
    combined_loss = coefficient * first_loss * weights + (
        1.0 - coefficient
    ) * second_loss * second_weights
    combined_weight = coefficient * weights + (1.0 - coefficient) * second_weights
    return combined_loss.sum() / combined_weight.sum().clamp_min(1.0e-8)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FashionMNISTTrainer:
    """Train, save, load, and evaluate the high-accuracy ensemble."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = self.default_data_root()
        self.batch_size = 256 if self.device.type == "cuda" else 96
        self.validation_ratio = 0.05
        self.selection_ratio = 0.05
        self.num_workers = min(4, max((os.cpu_count() or 2) // 2, 1))
        self.training_profile = "max"
        self.resume_training = True

        self._full_train_data: Optional[LocalFashionMNIST] = None
        self._train_indices: Optional[Tensor] = None
        self._selection_indices: Optional[Tensor] = None
        self._calibration_indices: Optional[Tensor] = None

        self.set_model()

    def default_data_root(self) -> Path:
        return REPO_ROOT / "data"

    @property
    def full_train_data(self) -> LocalFashionMNIST:
        if self._full_train_data is None:
            self._full_train_data = LocalFashionMNIST(self.data_root, split="train")
        return self._full_train_data

    def split_train_validation(self) -> None:
        if self._train_indices is None:
            train, selection, calibration = stratified_three_way_split(
                self.full_train_data.labels,
                selection_ratio=self.selection_ratio,
                calibration_ratio=self.validation_ratio,
                seed=2100,
            )
            self._train_indices = train
            self._selection_indices = selection
            self._calibration_indices = calibration

    @property
    def train_indices(self) -> Tensor:
        self.split_train_validation()
        return self._train_indices

    @property
    def selection_indices(self) -> Tensor:
        self.split_train_validation()
        return self._selection_indices

    @property
    def calibration_indices(self) -> Tensor:
        self.split_train_validation()
        return self._calibration_indices

    @property
    def train_data(self) -> IndexedDataset:
        return IndexedDataset(self.full_train_data, self.train_indices)

    @property
    def validation_data(self) -> IndexedDataset:
        return IndexedDataset(self.full_train_data, self.calibration_indices)

    @property
    def train_data_loader(self) -> DataLoader:
        return self._make_loader(self.train_indices, shuffle=True)

    @property
    def validation_data_loader(self) -> DataLoader:
        return self._make_loader(self.calibration_indices, shuffle=False)

    def _make_loader(
        self,
        indices: Tensor,
        shuffle: bool,
        sample_weights: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        return DataLoader(
            IndexedDataset(self.full_train_data, indices, sample_weights),
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.num_workers > 0,
        )

    def check_device(self) -> None:
        print(f"Running on {self.device}.")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def set_model(self) -> None:
        self.model = FashionEnsemble().to(self.device)
        initialize_module(self.model)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1.0e-3, weight_decay=5.0e-4
        )

    def configure_training(
        self,
        profile: str = "max",
        resume: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        if profile not in {"smoke", "balanced", "max"}:
            raise ValueError("profile must be smoke, balanced, or max.")
        self.training_profile = profile
        self.resume_training = bool(resume)
        if num_workers is not None:
            self.num_workers = max(int(num_workers), 0)

    def _profile(self) -> Dict[str, float | int]:
        profiles: Dict[str, Dict[str, float | int]] = {
            "smoke": {
                "base_epochs": 1,
                "fine_epochs": 1,
                "specialist_epochs": 1,
                "specialist_fine_epochs": 1,
                "patience": 2,
            },
            "balanced": {
                "base_epochs": 32,
                "fine_epochs": 4,
                "specialist_epochs": 28,
                "specialist_fine_epochs": 4,
                "patience": 8,
            },
            "max": {
                "base_epochs": 55,
                "fine_epochs": 7,
                "specialist_epochs": 45,
                "specialist_fine_epochs": 6,
                "patience": 11,
            },
        }
        return profiles[self.training_profile]

    def _checkpoint_directory(self) -> Path:
        directory = REPO_ROOT / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _augmenter_for_member(self, member_index: int, fine_tune: bool = False):
        configurations = [
            dict(degrees=7, translate=0.07, scale=0.05, erase_probability=0.10),
            dict(degrees=6, translate=0.06, scale=0.04, erase_probability=0.06),
            dict(degrees=8, translate=0.08, scale=0.06, erase_probability=0.12),
            dict(degrees=6, translate=0.07, scale=0.04, erase_probability=0.14),
            dict(degrees=8, translate=0.07, scale=0.05, erase_probability=0.10),
        ]
        config = configurations[member_index].copy()
        if fine_tune:
            config["degrees"] *= 0.65
            config["translate"] *= 0.65
            config["scale"] *= 0.65
            config["erase_probability"] *= 0.5
        return BatchAugmenter(**config)

    @torch.no_grad()
    def _evaluate_network(
        self,
        network: nn.Module,
        indices: Tensor,
        specialist: bool = False,
    ) -> Tuple[float, Tensor, Tensor]:
        network.eval()
        loader = self._make_loader(indices, shuffle=False)
        all_logits: List[Tensor] = []
        all_labels: List[Tensor] = []
        correct = 0
        total = 0

        for images, labels, _, _ in loader:
            images = normalize_images(
                images.to(self.device, non_blocking=True)
            )
            labels = labels.to(self.device, non_blocking=True)
            if specialist:
                labels = map_upper_labels(labels)
                valid = labels >= 0
                if not bool(valid.any()):
                    continue
                images = images[valid]
                labels = labels[valid]
            logits = network(images)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        accuracy = correct / max(total, 1)
        return accuracy, torch.cat(all_logits), torch.cat(all_labels)

    def _train_single_network(
        self,
        network: nn.Module,
        train_indices: Tensor,
        validation_indices: Optional[Tensor],
        sample_weights: Tensor,
        epochs: int,
        augmenter: BatchAugmenter,
        learning_rate: float,
        weight_decay: float,
        seed: int,
        patience: int,
        specialist: bool = False,
        mixup_fraction: float = 0.60,
    ) -> Tuple[Dict[str, Tensor], float]:
        set_global_seed(seed)
        network.to(self.device)
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loader = self._make_loader(
            train_indices,
            shuffle=True,
            sample_weights=sample_weights,
        )
        total_steps = max(epochs * len(loader), 1)
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=max(len(loader) * min(5, max(epochs // 5, 1)), 1),
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda"
        )
        ema = ExponentialMovingAverage(network, decay=0.9995)
        best_state = copy.deepcopy(network.state_dict())
        best_accuracy = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            network.train()
            total_loss = 0.0
            total_seen = 0
            total_correct = 0
            mixup_enabled = epoch <= max(int(epochs * mixup_fraction), 1)

            for images, labels, _, weights in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                weights = weights.to(self.device, dtype=torch.float32, non_blocking=True)

                if specialist:
                    mapped = map_upper_labels(labels)
                    valid = mapped >= 0
                    if not bool(valid.any()):
                        continue
                    images = images[valid]
                    labels = mapped[valid]
                    weights = weights[valid]

                images = augmenter(images)
                mixup_data = None
                if mixup_enabled and images.shape[0] > 1 and random.random() < 0.40:
                    beta = torch.distributions.Beta(0.15, 0.15)
                    coefficient = float(beta.sample().item())
                    permutation = torch.randperm(images.shape[0], device=self.device)
                    images = coefficient * images + (1.0 - coefficient) * images[permutation]
                    mixup_data = (labels[permutation], weights[permutation], coefficient)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    "cuda", enabled=self.device.type == "cuda"
                ):
                    logits = network(images)
                    loss = weighted_classification_loss(
                        logits,
                        labels,
                        weights,
                        label_smoothing=0.02,
                        mixup=mixup_data,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                ema.update(network)

                total_loss += float(loss.item()) * labels.numel()
                total_seen += int(labels.numel())
                if mixup_data is None:
                    total_correct += int((logits.argmax(dim=1) == labels).sum().item())

            ema_model = copy.deepcopy(network)
            ema.copy_to(ema_model)
            if validation_indices is not None:
                validation_accuracy, _, _ = self._evaluate_network(
                    ema_model, validation_indices, specialist=specialist
                )
            else:
                validation_accuracy = -1.0

            training_accuracy = total_correct / max(total_seen, 1)
            average_loss = total_loss / max(total_seen, 1)
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"loss {average_loss:.4f} | "
                f"train {training_accuracy * 100.0:.2f}% | "
                + (
                    f"validation {validation_accuracy * 100.0:.2f}%"
                    if validation_indices is not None
                    else "fine-tuning"
                )
            )

            if validation_indices is None:
                best_state = copy.deepcopy(ema_model.state_dict())
            elif validation_accuracy > best_accuracy + 1.0e-5:
                best_accuracy = validation_accuracy
                best_state = copy.deepcopy(ema_model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epoch >= 15 and epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break

            del ema_model

        network.load_state_dict(best_state)
        return best_state, best_accuracy

    @torch.no_grad()
    def _collect_member_logits(self, indices: Tensor) -> Tuple[Tensor, Tensor]:
        loader = self._make_loader(indices, shuffle=False)
        logits_parts: List[Tensor] = []
        label_parts: List[Tensor] = []
        self.model.eval()
        for images, labels, _, _ in loader:
            images = normalize_images(images.to(self.device, non_blocking=True))
            logits = self.model.member_logits(images)
            logits_parts.append(logits.detach().cpu())
            label_parts.append(labels.clone())
        return torch.cat(logits_parts), torch.cat(label_parts)

    def _fit_calibrator(self, member_logits: Tensor, labels: Tensor) -> float:
        logits = member_logits.to(self.device)
        labels = labels.to(self.device)
        member_count = logits.shape[1]
        raw_weights = nn.Parameter(torch.zeros(member_count, NUM_CLASSES, device=self.device))
        raw_temperatures = nn.Parameter(torch.zeros(member_count, device=self.device))
        bias = nn.Parameter(torch.zeros(NUM_CLASSES, device=self.device))
        optimizer = torch.optim.Adam(
            (raw_weights, raw_temperatures, bias), lr=0.05, weight_decay=0.0
        )

        best_accuracy = -1.0
        best_values = None
        for step in range(600):
            optimizer.zero_grad(set_to_none=True)
            weights = raw_weights.softmax(dim=0)
            temperatures = raw_temperatures.exp().clamp(0.35, 4.0)
            combined = (
                logits / temperatures.reshape(1, -1, 1)
                * weights.reshape(1, member_count, NUM_CLASSES)
            ).sum(dim=1) + bias
            loss = F.cross_entropy(combined, labels)
            loss = loss + 0.002 * raw_weights.square().mean()
            loss = loss + 0.001 * raw_temperatures.square().mean()
            loss.backward()
            optimizer.step()

            if step % 10 == 0 or step == 599:
                accuracy = float((combined.argmax(dim=1) == labels).float().mean().item())
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_values = (
                        weights.detach().clone(),
                        temperatures.detach().clone(),
                        bias.detach().clone(),
                    )

        if best_values is None:
            raise RuntimeError("Calibration failed to produce parameters.")
        weights, temperatures, bias = best_values
        self.model.class_weights.copy_(weights)
        self.model.temperatures.copy_(temperatures)
        self.model.class_bias.copy_(bias)
        return best_accuracy

    @torch.no_grad()
    def _hard_example_weights(
        self, indices: Tensor, member_logits: Tensor, labels: Tensor
    ) -> Tensor:
        combined = self.model.combine_member_logits(member_logits.to(self.device))
        probabilities = combined.softmax(dim=1)
        labels_device = labels.to(self.device)
        true_probability = probabilities.gather(1, labels_device.unsqueeze(1)).squeeze(1)
        entropy = -(
            probabilities * probabilities.clamp_min(1.0e-8).log()
        ).sum(dim=1) / math.log(NUM_CLASSES)
        member_probabilities = member_logits.to(self.device).softmax(dim=2)
        member_true = member_probabilities.gather(
            2,
            labels_device.reshape(-1, 1, 1).expand(-1, member_probabilities.shape[1], 1),
        ).squeeze(2)
        disagreement = member_true.std(dim=1)
        hard = 1.0 + 1.45 * (1.0 - true_probability) + 0.55 * entropy + 0.55 * disagreement
        hard = hard.clamp(1.0, 3.0).cpu()
        output = torch.ones(len(self.full_train_data), dtype=torch.float32)
        output[indices] = hard
        return output

    def _train_specialist(
        self,
        base_weights: Tensor,
        profile: Dict[str, float | int],
    ) -> None:
        train_upper = self.train_indices[
            torch.isin(self.full_train_data.labels[self.train_indices], torch.tensor(UPPER_CLASSES))
        ]
        selection_upper = self.selection_indices[
            torch.isin(
                self.full_train_data.labels[self.selection_indices],
                torch.tensor(UPPER_CLASSES),
            )
        ]
        full_upper = torch.cat((train_upper, selection_upper))
        checkpoint = self._checkpoint_directory() / "specialist_final.pth"
        if self.resume_training and checkpoint.exists():
            self.model.specialist.load_state_dict(
                torch.load(checkpoint, map_location=self.device, weights_only=True)
            )
            print("Loaded specialist checkpoint.")
            return

        print("\nTraining upper-body specialist (0, 2, 4, 6).")
        bootstrap = bayesian_bootstrap_weights(self.full_train_data.labels, seed=96001)
        weights = bootstrap * base_weights
        state, _ = self._train_single_network(
            self.model.specialist,
            train_upper,
            selection_upper,
            weights,
            epochs=int(profile["specialist_epochs"]),
            augmenter=BatchAugmenter(
                degrees=6,
                translate=0.06,
                scale=0.04,
                erase_probability=0.06,
            ),
            learning_rate=2.0e-3,
            weight_decay=5.0e-4,
            seed=96001,
            patience=int(profile["patience"]),
            specialist=True,
        )
        self.model.specialist.load_state_dict(state)
        fine_state, _ = self._train_single_network(
            self.model.specialist,
            full_upper,
            None,
            weights,
            epochs=int(profile["specialist_fine_epochs"]),
            augmenter=BatchAugmenter(
                degrees=4,
                translate=0.04,
                scale=0.03,
                erase_probability=0.03,
            ),
            learning_rate=3.0e-4,
            weight_decay=3.0e-4,
            seed=97001,
            patience=999,
            specialist=True,
            mixup_fraction=0.35,
        )
        self.model.specialist.load_state_dict(fine_state)
        torch.save(self.model.specialist.state_dict(), checkpoint)

    @torch.no_grad()
    def _collect_specialist_probabilities(self, indices: Tensor) -> Tensor:
        loader = self._make_loader(indices, shuffle=False)
        parts: List[Tensor] = []
        self.model.specialist.eval()
        for images, _, _, _ in loader:
            images = normalize_images(images.to(self.device, non_blocking=True))
            parts.append(self.model.specialist(images).softmax(dim=1).cpu())
        return torch.cat(parts)

    def _calibrate_specialist_gate(
        self,
        general_probabilities: Tensor,
        specialist_probabilities: Tensor,
        labels: Tensor,
    ) -> float:
        upper = torch.tensor(UPPER_CLASSES, dtype=torch.long)
        upper_probabilities = general_probabilities[:, upper]
        upper_mass = upper_probabilities.sum(dim=1)
        top_two = general_probabilities.topk(2, dim=1).values
        margins = top_two[:, 0] - top_two[:, 1]
        top_labels = general_probabilities.argmax(dim=1)
        top_is_upper = (top_labels.unsqueeze(1) == upper.unsqueeze(0)).any(dim=1)

        best_accuracy = float((top_labels == labels).float().mean().item())
        best_parameters = (0.55, 0.35, 0.65)
        for threshold in (0.40, 0.50, 0.60, 0.70, 0.80):
            for margin_limit in (0.08, 0.15, 0.25, 0.40, 0.65, 1.00):
                trigger = (
                    (upper_mass >= threshold)
                    & (margins <= margin_limit)
                    & top_is_upper
                )
                if not bool(trigger.any()):
                    continue
                for alpha in (0.25, 0.45, 0.65, 0.80, 1.00):
                    output = general_probabilities.clone()
                    conditional = upper_probabilities[trigger] / upper_mass[
                        trigger
                    ].unsqueeze(1).clamp_min(1.0e-8)
                    blended = (
                        (1.0 - alpha) * conditional
                        + alpha * specialist_probabilities[trigger]
                    )
                    rows = torch.nonzero(trigger, as_tuple=False).flatten()
                    output[rows.unsqueeze(1), upper.unsqueeze(0)] = (
                        blended * upper_mass[trigger].unsqueeze(1)
                    )
                    accuracy = float(
                        (output.argmax(dim=1) == labels).float().mean().item()
                    )
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_parameters = (threshold, margin_limit, alpha)

        threshold, margin_limit, alpha = best_parameters
        self.model.specialist_threshold.fill_(threshold)
        self.model.specialist_margin.fill_(margin_limit)
        self.model.specialist_alpha.fill_(alpha)
        return best_accuracy

    @torch.no_grad()
    def _probabilities_from_member_logits(self, member_logits: Tensor) -> Tensor:
        return self.model.combine_member_logits(member_logits.to(self.device)).softmax(dim=1).cpu()

    @torch.no_grad()
    def _calibrate_tta(self, indices: Tensor, labels: Tensor) -> float:
        loader = self._make_loader(indices, shuffle=False)
        original_parts: List[Tensor] = []
        averaged_parts: List[Tensor] = []
        self.model.eval()
        for images, _, _, _ in loader:
            images = normalize_images(images.to(self.device, non_blocking=True))
            original = self.model.predict_proba_once(images)
            variants = [
                torch.flip(images, dims=(3,)),
                self.model._shift(images, horizontal=-1),
                self.model._shift(images, horizontal=1),
                self.model._shift(images, vertical=-1),
                self.model._shift(images, vertical=1),
            ]
            averaged = original.clone()
            for variant in variants:
                averaged += self.model.predict_proba_once(variant)
            averaged /= float(len(variants) + 1)
            original_parts.append(original.cpu())
            averaged_parts.append(averaged.cpu())

        original = torch.cat(original_parts)
        averaged = torch.cat(averaged_parts)
        top_two = original.topk(2, dim=1).values
        confidence = top_two[:, 0]
        margin = top_two[:, 0] - top_two[:, 1]
        best_accuracy = float((original.argmax(dim=1) == labels).float().mean().item())
        best_parameters = (0.93, 0.35)

        for confidence_limit in (0.70, 0.80, 0.88, 0.93, 0.97, 1.01):
            for margin_limit in (0.05, 0.12, 0.20, 0.35, 0.55, 1.01):
                use_tta = (confidence < confidence_limit) | (margin < margin_limit)
                output = original.clone()
                output[use_tta] = averaged[use_tta]
                accuracy = float((output.argmax(dim=1) == labels).float().mean().item())
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_parameters = (confidence_limit, margin_limit)

        confidence_limit, margin_limit = best_parameters
        self.model.tta_confidence.fill_(confidence_limit)
        self.model.tta_margin.fill_(margin_limit)
        return best_accuracy

    def train(self) -> None:
        """Train the complete ensemble using only the official train split."""

        if self.device.type != "cuda":
            print(
                "Warning: this training recipe is very slow on CPU; "
                "a single GPU is strongly recommended."
            )
        set_global_seed(2100)
        profile = self._profile()
        self.split_train_validation()
        print(
            f"Data split: train={len(self.train_indices)}, "
            f"selection={len(self.selection_indices)}, "
            f"calibration={len(self.calibration_indices)}"
        )

        labels = self.full_train_data.labels
        provisional_member_logits: List[Tensor] = []
        provisional_labels = None

        # Phase 1: independently train and select each architecture.
        for member_index, member in enumerate(self.model.members):
            checkpoint = self._checkpoint_directory() / f"member_{member_index}_selected.pth"
            if self.resume_training and checkpoint.exists():
                member.load_state_dict(
                    torch.load(checkpoint, map_location=self.device, weights_only=True)
                )
                print(f"Loaded selected checkpoint for member {member_index + 1}.")
            else:
                print(f"\nTraining generalist {member_index + 1}/{len(self.model.members)}.")
                bootstrap = bayesian_bootstrap_weights(
                    labels, seed=3100 + member_index * 101
                )
                state, best_accuracy = self._train_single_network(
                    member,
                    self.train_indices,
                    self.selection_indices,
                    bootstrap,
                    epochs=int(profile["base_epochs"]),
                    augmenter=self._augmenter_for_member(member_index),
                    learning_rate=2.2e-3,
                    weight_decay=5.0e-4,
                    seed=4100 + member_index * 101,
                    patience=int(profile["patience"]),
                )
                member.load_state_dict(state)
                torch.save(member.state_dict(), checkpoint)
                print(
                    f"Generalist {member_index + 1} best selection accuracy: "
                    f"{best_accuracy * 100.0:.2f}%"
                )

        # Provisional calibration on the model-selection subset.
        selection_logits, selection_labels = self._collect_member_logits(
            self.selection_indices
        )
        provisional_accuracy = self._fit_calibrator(selection_logits, selection_labels)
        print(
            f"Provisional calibrated selection accuracy: "
            f"{provisional_accuracy * 100.0:.2f}%"
        )

        # Find examples the diverse ensemble considers difficult.
        fine_indices = torch.cat((self.train_indices, self.selection_indices))
        fine_logits, fine_labels = self._collect_member_logits(fine_indices)
        hard_weights = self._hard_example_weights(fine_indices, fine_logits, fine_labels)

        # Phase 2: low-rate full 95% fine-tuning with hard-example and
        # Bayesian-bootstrap weighting.  The final 5% remains untouched for
        # calibration and honest local evaluation.
        for member_index, member in enumerate(self.model.members):
            checkpoint = self._checkpoint_directory() / f"member_{member_index}_final.pth"
            if self.resume_training and checkpoint.exists():
                member.load_state_dict(
                    torch.load(checkpoint, map_location=self.device, weights_only=True)
                )
                print(f"Loaded final checkpoint for member {member_index + 1}.")
                continue

            print(f"\nFine-tuning generalist {member_index + 1} on 95% of training data.")
            bootstrap = bayesian_bootstrap_weights(
                labels, seed=7100 + member_index * 101
            )
            weights = bootstrap * hard_weights
            state, _ = self._train_single_network(
                member,
                fine_indices,
                None,
                weights,
                epochs=int(profile["fine_epochs"]),
                augmenter=self._augmenter_for_member(member_index, fine_tune=True),
                learning_rate=3.0e-4,
                weight_decay=3.0e-4,
                seed=8100 + member_index * 101,
                patience=999,
                mixup_fraction=0.35,
            )
            member.load_state_dict(state)
            torch.save(member.state_dict(), checkpoint)

        # Train and fine-tune the specialist using the same untouched
        # calibration policy.
        self._train_specialist(hard_weights, profile)

        # Final class-wise stacking and temperature calibration.
        calibration_logits, calibration_labels = self._collect_member_logits(
            self.calibration_indices
        )
        calibrated_accuracy = self._fit_calibrator(
            calibration_logits, calibration_labels
        )
        print(
            f"Generalist calibration accuracy: {calibrated_accuracy * 100.0:.2f}%"
        )

        general_probabilities = self._probabilities_from_member_logits(
            calibration_logits
        )
        specialist_probabilities = self._collect_specialist_probabilities(
            self.calibration_indices
        )
        specialist_accuracy = self._calibrate_specialist_gate(
            general_probabilities,
            specialist_probabilities,
            calibration_labels,
        )
        print(
            f"After specialist gate calibration: {specialist_accuracy * 100.0:.2f}%"
        )

        tta_accuracy = self._calibrate_tta(
            self.calibration_indices, calibration_labels
        )
        print(f"After selective TTA calibration: {tta_accuracy * 100.0:.2f}%")

        self.export_model(DEFAULT_WEIGHTS)
        print(f"Saved final weights to {DEFAULT_WEIGHTS}")

    def _prepare_image(self, image: Tensor) -> Tensor:
        """Convert one supplied tensor to normalized 1 x 1 x 28 x 28 form."""

        if not torch.is_tensor(image):
            raise TypeError("image must be a torch.Tensor.")
        image = image.detach()

        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:
            if image.shape[0] == 1:
                image = image.unsqueeze(0)
            elif image.shape[-1] == 1:
                image = image.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError("A single grayscale image is required.")
        elif image.ndim == 4:
            if image.shape[0] != 1 or image.shape[1] != 1:
                raise ValueError("A single grayscale image is required.")
        else:
            raise ValueError("image must have 2, 3, or 4 dimensions.")

        if tuple(image.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("image must have size 28 x 28.")

        original_dtype = image.dtype
        image = image.to(torch.float32)
        minimum = float(image.min().item())
        maximum = float(image.max().item())

        if not original_dtype.is_floating_point or maximum > 5.0:
            image = image / 255.0
            image = normalize_images(image)
        elif minimum >= -1.0e-6 and maximum <= 1.0 + 1.0e-6:
            image = normalize_images(image)
        else:
            # Values outside [0, 1] but within a small numeric range are
            # assumed to have already been normalized by the caller.
            pass

        return image.to(self.device)

    @torch.inference_mode()
    def eval(self, image: Tensor) -> int:
        """Return the predicted integer label for one 28 x 28 tensor."""

        self.model.eval()
        prepared = self._prepare_image(image)
        probabilities = self.model.predict_proba(prepared, use_tta=True)
        return int(probabilities.argmax(dim=1).item())

    @torch.inference_mode()
    def validate_all(self) -> None:
        self.model.eval()
        loader = self._make_loader(self.calibration_indices, shuffle=False)
        correct = 0
        total = 0
        start = time.perf_counter()
        for images, labels, _, _ in loader:
            images = normalize_images(images.to(self.device, non_blocking=True))
            labels = labels.to(self.device, non_blocking=True)
            probabilities = self.model.predict_proba(images, use_tta=True)
            correct += int((probabilities.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
        elapsed = time.perf_counter() - start
        print(f"Validation Accuracy: {correct / max(total, 1) * 100.0:.2f}%")
        print(f"Validation inference time: {elapsed:.2f} seconds for {total} images")

    def sample_validation_image(self):
        index = int(
            self.calibration_indices[
                torch.randint(self.calibration_indices.numel(), (1,)).item()
            ].item()
        )
        return self.full_train_data[index]

    def export_model(self, file_name: str | Path = DEFAULT_WEIGHTS) -> None:
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), file_name)

    def import_model(self, file_name: str | Path = DEFAULT_WEIGHTS) -> None:
        state_dict = torch.load(
            file_name,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Fashion-MNIST ensemble.")
    parser.add_argument(
        "--profile",
        choices=("smoke", "balanced", "max"),
        default="max",
        help="smoke checks the pipeline; max performs the strongest training.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints/ and train again.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override DataLoader worker count.",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_WEIGHTS), help="Final weight-file path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    trainer = FashionMNISTTrainer()
    trainer.configure_training(
        profile=arguments.profile,
        resume=not arguments.no_resume,
        num_workers=arguments.workers,
    )
    trainer.check_device()
    trainer.train()

    if Path(arguments.output) != DEFAULT_WEIGHTS:
        trainer.export_model(arguments.output)
        print(f"Also saved weights to {arguments.output}")

    trainer.validate_all()
    sample, label = trainer.sample_validation_image()
    prediction = trainer.eval(sample)
    print(f"Label: {label}")
    print(f"Prediction: {prediction}")

    trainer.import_model(arguments.output)
    print("Reload test passed.")
