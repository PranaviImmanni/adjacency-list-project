"""Fashion-MNIST: checksum-verified download, IDX parsing, fixed split, normalization.

No torchvision: the four official IDX files are downloaded directly and
checked against the same MD5s torchvision uses, so the lab's CUDA build of
PyTorch is never touched by an extra install.

Split: the 60,000 training images are divided once, with a dedicated
generator seeded by `split_seed` (independent of the model seeds), into
train (60,000 - val_size) and validation (val_size). The 10,000 test images
are the official test set and are only used for reporting.

Normalization: pixels are scaled to [0, 1], then standardized with the mean
and standard deviation of the training split only (computed in float64).
"""

import gzip
import hashlib
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

from src.gcn.config import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "FashionMNIST" / "raw"
MIRRORS = (
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
)
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
}
CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
IDX_DTYPE_UINT8 = 0x08


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def download(raw_dir=RAW_DIR):
    """Make sure the four IDX files exist with the expected checksums."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.values():
        path = raw_dir / name
        if path.exists() and _md5(path) == md5:
            continue
        errors = []
        for mirror in MIRRORS:
            try:
                urllib.request.urlretrieve(mirror + name, path)
            except OSError as error:
                errors.append(f"{mirror}: {error}")
                continue
            if _md5(path) == md5:
                break
            errors.append(f"{mirror}: checksum mismatch")
        else:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"could not download {name}: {'; '.join(errors)}")
    return raw_dir


def parse_idx(data):
    """Parse an (uncompressed) IDX byte string of unsigned bytes into a uint8 tensor."""
    zero, dtype_code, ndim = struct.unpack(">HBB", data[:4])
    if zero != 0 or dtype_code != IDX_DTYPE_UINT8:
        raise ValueError(f"not an unsigned-byte IDX file (header {data[:4].hex()})")
    shape = struct.unpack(f">{ndim}I", data[4:4 + 4 * ndim])
    body = data[4 + 4 * ndim:]
    expected = 1
    for size in shape:
        expected *= size
    if len(body) != expected:
        raise ValueError(f"IDX body has {len(body)} bytes, header says {expected}")
    return torch.frombuffer(bytearray(body), dtype=torch.uint8).reshape(shape)


def read_idx_gz(path):
    with gzip.open(path, "rb") as f:
        return parse_idx(f.read())


@dataclass
class ImageData:
    x_train: torch.Tensor  # [n, 784] float32, standardized
    y_train: torch.Tensor  # [n] int64
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    mean: float
    std: float
    split_seed: int
    val_indices_sha256: str
    num_features: int = 784
    num_classes: int = 10

    def summary(self):
        return {
            "dataset": "fashion_mnist",
            "num_train": len(self.y_train), "num_val": len(self.y_val), "num_test": len(self.y_test),
            "num_features": self.num_features, "num_classes": self.num_classes,
            "split_seed": self.split_seed, "val_indices_sha256": self.val_indices_sha256,
            "normalization": {"mean": self.mean, "std": self.std, "computed_on": "training split"},
        }

    def eval_set(self):
        """Validation then test samples, with masks over the concatenation.

        The masks let the GCN error-analysis code (written for node masks) work
        on samples unchanged: "all" means validation + test.
        """
        x = torch.cat([self.x_val, self.x_test])
        y = torch.cat([self.y_val, self.y_test])
        val_mask = torch.zeros(len(y), dtype=torch.bool)
        val_mask[:len(self.y_val)] = True
        return x, y, val_mask, ~val_mask


def split_and_normalize(images, labels, test_images, test_labels, val_size, split_seed):
    """Fixed train/val split and train-split standardization. Inputs are uint8 tensors."""
    n = len(labels)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed))
    val_idx, train_idx = order[:val_size].sort().values, order[val_size:].sort().values

    def flat(t):
        return t.reshape(len(t), -1).to(torch.float64) / 255.0

    train = flat(images[train_idx])
    mean, std = train.mean().item(), train.std().item()

    def norm(t):
        return ((t - mean) / std).to(torch.float32)

    digest = hashlib.sha256(val_idx.numpy().tobytes()).hexdigest()[:16]
    return ImageData(
        x_train=norm(train), y_train=labels[train_idx].long(),
        x_val=norm(flat(images[val_idx])), y_val=labels[val_idx].long(),
        x_test=norm(flat(test_images)), y_test=test_labels.long(),
        mean=mean, std=std, split_seed=split_seed, val_indices_sha256=digest,
        num_features=train.shape[1],
    )


def load_fashion_mnist(val_size=5000, split_seed=0, raw_dir=RAW_DIR):
    raw_dir = download(raw_dir)
    tensors = {key: read_idx_gz(raw_dir / name) for key, (name, _) in FILES.items()}
    return split_and_normalize(tensors["train_images"], tensors["train_labels"],
                               tensors["test_images"], tensors["test_labels"], val_size, split_seed)
