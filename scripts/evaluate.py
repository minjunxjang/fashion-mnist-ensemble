"""Evaluate trained weights on the Fashion-MNIST test split (10,000 images).

    python scripts/evaluate.py --weights weights/fashion_ensemble.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fashion_ensemble.model import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_WEIGHTS,
    FashionMNISTTrainer,
    LocalFashionMNIST,
    normalize_images,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="evaluate first N images only")
    args = parser.parse_args()

    trainer = FashionMNISTTrainer()
    trainer.import_model(args.weights)
    model, device = trainer.model, trainer.device

    test = LocalFashionMNIST(trainer.data_root, split="test")
    if args.limit:
        test = torch.utils.data.Subset(test, range(args.limit))
    loader = DataLoader(test, batch_size=256, shuffle=False)

    confusion = torch.zeros(10, 10, dtype=torch.long)
    start = time.perf_counter()
    for images, labels in loader:
        images = normalize_images(images.to(device))
        preds = model.predict_proba(images, use_tta=not args.no_tta).argmax(dim=1).cpu()
        for t, p in zip(labels.tolist(), preds.tolist()):
            confusion[t, p] += 1
    elapsed = time.perf_counter() - start

    total = int(confusion.sum())
    acc = confusion.diag().sum().item() / total
    print(f"test accuracy: {acc:.4f}  ({confusion.diag().sum().item()}/{total}, {elapsed:.1f}s)")
    print("\nper-class accuracy")
    for i, name in enumerate(CLASS_NAMES):
        row = confusion[i].sum().item()
        print(f"  {name:<12} {confusion[i, i].item() / max(row, 1):.4f}")
    print("\nconfusion matrix (rows = true, cols = predicted)")
    print("     " + " ".join(f"{i:>5}" for i in range(10)))
    for i in range(10):
        print(f"{i:>4} " + " ".join(f"{v:>5}" for v in confusion[i].tolist()))


if __name__ == "__main__":
    main()
