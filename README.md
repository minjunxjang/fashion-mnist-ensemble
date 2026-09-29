# Fashion-MNIST Heterogeneous CNN Ensemble

**95.10 % test accuracy** on the official Fashion-MNIST test split (10,000 images),
trained from scratch on the 60k training split only, with no external data or
pre-trained weights.

The idea: instead of one big network, combine five small CNNs with deliberately
different inductive biases, calibrate how much to trust each one *per class*,
and send the classes that everyone confuses to a dedicated specialist.

## Architecture

| Member | Idea | Params |
|---|---|---:|
| Wide SE-ResNet | pre-activation residual blocks + squeeze-excitation, late downsampling, avg‖max pooling | 1.80 M |
| Edge-aware SE-ResNet | fixed Sobel filters add horizontal/vertical edge channels to the input | 1.41 M |
| Multi-scale net | parallel 3×3 / 5×5 / dilated-3×3 branches in every block | 0.97 M |
| Spatial VGG | plain VGG stack that keeps a 4×4 spatial map before the classifier | 1.73 M |
| Tiny ConvNeXt | 7×7 depthwise conv, LayerNorm, layer scale, stochastic depth | 0.93 M |
| Upper-body specialist | Wide SE-ResNet trained only on T-shirt / Pullover / Coat / Shirt | 1.80 M |
| **Total** | | **8.63 M** |

### Combining the members
1. **Class-wise stacking + temperature scaling.** Each member gets its own
   temperature and a separate weight for each of the 10 classes (softmax over
   members), plus a class bias. These 5×10 + 5 + 10 parameters are fit by
   minimizing cross-entropy on a held-out calibration set.
2. **Gated specialist.** When the ensemble's top class is an upper-body class,
   the upper-body probability mass is high, and the top-2 margin is small, the
   probabilities *within* those four classes are blended with the specialist's.
   The threshold, margin and blend weight are grid-searched on the calibration set.
3. **Selective TTA.** Flip and ±1 px shifts are averaged only for low-confidence
   inputs (thresholds also calibrated).

## Training pipeline

- **Stratified 90 / 5 / 5 split** of the training set: *train* / *model selection*
  (early stopping, checkpoint choice) / *calibration* (stacking weights, gates).
  The calibration 5 % is never trained on, so calibration isn't fit to memorized data.
- **Bayesian bootstrap.** Each member sees the data with its own random
  Exp(1) sample weights, adding diversity on top of architecture differences.
- **Phase 1:** train each member independently. AdamW, warmup + cosine LR,
  EMA of weights, AMP, label smoothing 0.02, MixUp for the first 60 % of epochs,
  GPU batch augmentation (small affine, flip, brightness/contrast, random erasing).
- **Hard-example mining.** Using the phase-1 ensemble, each sample is reweighted
  by (1 − p_true), predictive entropy, and member disagreement.
- **Phase 2:** low-LR fine-tuning of every member on train + selection (95 %)
  with the hard-example × bootstrap weights.
- **Specialist → calibration → gate search → TTA search**, all on the untouched 5 %.

Training profiles: `smoke` (pipeline check), `balanced`, `max` (reported result).
Every stage checkpoints to `checkpoints/` and resumes automatically.

## Results

Official test split, 10,000 images:

| Setting | Accuracy |
|---|---:|
| Full ensemble + selective TTA | **95.10 %** |
| Full ensemble, no TTA | 95.15 % |

Selective TTA does not help on the test set (−0.05 pp, within noise): the
gains it showed on the 3,000-image calibration set did not transfer.

| Class | Acc. | | Class | Acc. |
|---|---:|---|---|---:|
| T-shirt/top | 91.9 % | | Sandal | 99.0 % |
| Trouser | 99.2 % | | Shirt | **83.0 %** |
| Pullover | 94.0 % | | Sneaker | 98.7 % |
| Dress | 96.4 % | | Bag | 99.4 % |
| Coat | 92.0 % | | Ankle boot | 97.4 % |

Most remaining errors are Shirt ↔ T-shirt/top (81 + 55 of 490 errors),
which is exactly the group the specialist targets. It is also the pair that is
genuinely ambiguous at 28×28.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py                 # Fashion-MNIST IDX files -> data/
python scripts/evaluate.py                      # accuracy, per-class, confusion matrix (~2 min on CPU)

# train from scratch (a single GPU is strongly recommended)
python -m src.fashion_ensemble.model --profile smoke    # pipeline check
python -m src.fashion_ensemble.model --profile max
```

```python
from src.fashion_ensemble import FashionMNISTTrainer
clf = FashionMNISTTrainer()
clf.import_model("weights/fashion_ensemble.pth")
label = clf.eval(image_28x28_tensor)   # int in [0, 9]
```

## Repository layout

```
src/fashion_ensemble/model.py   data loader, members, ensemble, trainer
scripts/download_data.py        fetch Fashion-MNIST
scripts/evaluate.py             test-split evaluation
weights/fashion_ensemble.pth    trained ensemble (33 MiB)
```

Dataset: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) © Zalando SE, MIT License.
