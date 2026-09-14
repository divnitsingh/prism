"""A small convolutional network written directly in NumPy.

The network is deliberately dependency-free. Everything a deep-learning
framework would normally provide - convolution, pooling, dropout, softmax
cross-entropy and the Adam optimiser - is implemented here so the app can
train a genuine CNN in the browser session without shipping a multi-hundred
megabyte runtime.

Architecture (mirrors a classic small-image classifier):

    input  1 x 8 x 8
      -> conv 3x3 (pad 1) -> ReLU -> maxpool 2x2      -> C1 x 4 x 4
      -> conv 3x3 (pad 1) -> ReLU -> maxpool 2x2      -> C2 x 2 x 2
      -> flatten -> dense -> ReLU -> dropout
      -> dense -> softmax over 10 classes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

__all__ = [
    "TrainConfig",
    "ConvNet",
    "train",
    "EpochRecord",
    "im2col",
    "col2im",
]


# --------------------------------------------------------------------------
# low level helpers
# --------------------------------------------------------------------------
def im2col(x: np.ndarray, kh: int, kw: int, pad: int, stride: int = 1) -> np.ndarray:
    """Flatten sliding windows into rows so convolution becomes a matmul."""
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w + 2 * pad - kw) // stride + 1

    cols = np.empty((n, c, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_end = i + stride * out_h
        for j in range(kw):
            j_end = j + stride * out_w
            cols[:, :, i, j, :, :] = xp[:, :, i:i_end:stride, j:j_end:stride]

    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)


def col2im(
    cols: np.ndarray,
    x_shape: tuple[int, int, int, int],
    kh: int,
    kw: int,
    pad: int,
    stride: int = 1,
) -> np.ndarray:
    """Inverse of :func:`im2col`, accumulating overlapping gradients."""
    n, c, h, w = x_shape
    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w + 2 * pad - kw) // stride + 1

    reshaped = cols.reshape(n, out_h, out_w, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        i_end = i + stride * out_h
        for j in range(kw):
            j_end = j + stride * out_w
            xp[:, :, i:i_end:stride, j:j_end:stride] += reshaped[:, :, i, j, :, :]

    if pad == 0:
        return xp
    return xp[:, :, pad : pad + h, pad : pad + w]


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
class Conv2D:
    """2-D convolution with 'same' padding for odd kernels."""

    def __init__(self, in_ch: int, out_ch: int, k: int, rng: np.random.Generator):
        fan_in = in_ch * k * k
        self.W = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(out_ch, in_ch, k, k))
        self.b = np.zeros(out_ch)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.k = k
        self.pad = k // 2
        self._cache: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        out_ch = self.W.shape[0]
        out_h = h + 2 * self.pad - self.k + 1
        out_w = w + 2 * self.pad - self.k + 1

        cols = im2col(x, self.k, self.k, self.pad)
        w_col = self.W.reshape(out_ch, -1)
        out = cols @ w_col.T + self.b
        self._cache = (x.shape, cols, w_col)
        return out.reshape(n, out_h, out_w, out_ch).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x_shape, cols, w_col = self._cache
        out_ch = self.W.shape[0]
        dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, out_ch)

        self.dW = (dout_col.T @ cols).reshape(self.W.shape)
        self.db = dout_col.sum(axis=0)

        dcols = dout_col @ w_col
        return col2im(dcols, x_shape, self.k, self.k, self.pad)

    def params(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class MaxPool2:
    """Non-overlapping 2x2 max pooling."""

    def __init__(self):
        self._cache: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError(f"MaxPool2 needs even spatial dims, received {h}x{w}.")
        grid = x.reshape(n, c, h // 2, 2, w // 2, 2)
        out = grid.max(axis=(3, 5))
        mask = grid == out[:, :, :, None, :, None]
        self._cache = (mask, x.shape)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        mask, x_shape = self._cache
        n, c, h, w = x_shape
        # Share the gradient evenly whenever a window has tied maxima.
        share = mask / mask.sum(axis=(3, 5), keepdims=True)
        grad = share * dout[:, :, :, None, :, None]
        return grad.reshape(n, c, h, w)

    def params(self):
        return []


class Flatten:
    def __init__(self):
        self._shape: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._shape)

    def params(self):
        return []


class Dense:
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class ReLU:
    def __init__(self):
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask

    def params(self):
        return []


class Dropout:
    """Inverted dropout - a no-op outside training."""

    def __init__(self, rate: float, rng: np.random.Generator):
        self.rate = float(np.clip(rate, 0.0, 0.95))
        self.rng = rng
        self.training = True
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        self._mask = (self.rng.random(x.shape) < keep) / keep
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0.0:
            return dout
        return dout * self._mask

    def params(self):
        return []


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Return mean loss plus the gradient with respect to the logits."""
    probs = softmax(logits)
    n = logits.shape[0]
    loss = float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, None)).mean())
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    return loss, grad / n


# --------------------------------------------------------------------------
# optimiser
# --------------------------------------------------------------------------
class Adam:
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def step(self, layers) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t

        for layer in layers:
            for name, param, grad in layer.params():
                key = id(param)
                if key not in self._m:
                    self._m[key] = np.zeros_like(param)
                    self._v[key] = np.zeros_like(param)

                m = self._m[key]
                v = self._v[key]
                m *= self.beta1
                m += (1.0 - self.beta1) * grad
                v *= self.beta2
                v += (1.0 - self.beta2) * (grad * grad)

                param -= self.lr * (m / bias1) / (np.sqrt(v / bias2) + self.eps)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    conv1_filters: int = 16
    conv2_filters: int = 32
    dense_units: int = 64
    dropout: float = 0.25
    learning_rate: float = 3e-3
    batch_size: int = 64
    epochs: int = 20
    seed: int = 42
    patience: int = 6


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


class ConvNet:
    """The full network, assembled from the layers above."""

    def __init__(self, config: TrainConfig, num_classes: int = 10, image_size: int = 8):
        if image_size % 4:
            raise ValueError("image_size must be divisible by 4 for two pooling stages.")

        self.config = config
        self.num_classes = num_classes
        self.image_size = image_size
        rng = np.random.default_rng(config.seed)
        self.rng = rng

        pooled = image_size // 4
        flat_dim = config.conv2_filters * pooled * pooled

        self.conv1 = Conv2D(1, config.conv1_filters, 3, rng)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2()
        self.conv2 = Conv2D(config.conv1_filters, config.conv2_filters, 3, rng)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2()
        self.flatten = Flatten()
        self.fc1 = Dense(flat_dim, config.dense_units, rng)
        self.relu3 = ReLU()
        self.drop = Dropout(config.dropout, rng)
        self.fc2 = Dense(config.dense_units, num_classes, rng)

        self.layers = [
            self.conv1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.relu2,
            self.pool2,
            self.flatten,
            self.fc1,
            self.relu3,
            self.drop,
            self.fc2,
        ]

    # -- inference ---------------------------------------------------------
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self.drop.training = training
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout: np.ndarray) -> None:
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict_proba(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        chunks = [
            softmax(self.forward(x[i : i + batch_size], training=False))
            for i in range(0, len(x), batch_size)
        ]
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, self.num_classes))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    # -- introspection -----------------------------------------------------
    def parameter_count(self) -> int:
        return int(sum(p.size for p in self._parameters()))

    def _parameters(self) -> list[np.ndarray]:
        return [p for layer in self.layers for _name, p, _g in layer.params()]

    def first_layer_kernels(self) -> np.ndarray:
        """Return the learned 3x3 filters of the first conv layer."""
        return self.conv1.W[:, 0, :, :].copy()

    def feature_maps(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Activations for a single image, keyed by stage name."""
        x = image.reshape(1, 1, self.image_size, self.image_size)
        a1 = self.relu1.forward(self.conv1.forward(x))
        p1 = self.pool1.forward(a1)
        a2 = self.relu2.forward(self.conv2.forward(p1))
        p2 = self.pool2.forward(a2)
        return {
            "conv1": a1[0].copy(),
            "pool1": p1[0].copy(),
            "conv2": a2[0].copy(),
            "pool2": p2[0].copy(),
        }

    def snapshot(self) -> list[np.ndarray]:
        return [p.copy() for p in self._parameters()]

    def restore(self, snapshot: list[np.ndarray]) -> None:
        params = self._parameters()
        if len(params) != len(snapshot):
            raise ValueError("Snapshot does not match the current architecture.")
        for target, saved in zip(params, snapshot):
            target[...] = saved


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------
def _iterate_batches(
    x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    order = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def _evaluate(model: ConvNet, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    logits_chunks = []
    for start in range(0, len(x), 512):
        logits_chunks.append(model.forward(x[start : start + 512], training=False))
    logits = np.concatenate(logits_chunks, axis=0)
    loss, _ = softmax_cross_entropy(logits, y)
    accuracy = float((logits.argmax(axis=1) == y).mean())
    return loss, accuracy


def train(
    model: ConvNet,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig | None = None,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> tuple[list[EpochRecord], int]:
    """Train with early stopping, restoring the best validation checkpoint.

    ``on_epoch`` is called after every epoch, which lets the interface stream
    the learning curves while training is still running.
    """
    config = config or model.config
    rng = np.random.default_rng(config.seed + 1)
    optimiser = Adam(config.learning_rate)

    history: list[EpochRecord] = []
    best_val = np.inf
    best_epoch = 0
    best_state = model.snapshot()
    stale = 0

    for epoch in range(1, config.epochs + 1):
        running_loss = 0.0
        correct = 0
        seen = 0

        for xb, yb in _iterate_batches(x_train, y_train, config.batch_size, rng):
            logits = model.forward(xb, training=True)
            loss, dlogits = softmax_cross_entropy(logits, yb)
            model.backward(dlogits)
            optimiser.step(model.layers)

            running_loss += loss * len(yb)
            correct += int((logits.argmax(axis=1) == yb).sum())
            seen += len(yb)

        train_loss = running_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)
        val_loss, val_acc = _evaluate(model, x_val, y_val)

        record = EpochRecord(epoch, train_loss, train_acc, val_loss, val_acc)
        history.append(record)
        if on_epoch is not None:
            on_epoch(record)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = model.snapshot()
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.restore(best_state)
    return history, best_epoch
