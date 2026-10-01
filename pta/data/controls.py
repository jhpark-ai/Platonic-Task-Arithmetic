"""The two control sets every edit is scored on: CIFAR-100 and ImageNet, 10 per class.

Both draws are seeded at 42 and identical for every model and every route, so a
control column measures what an edit does to images of tasks it never saw.

ImageNet is read from the parquet copy of the 50,000 validation images published on
the Hugging Face hub as `mrm8488/ImageNet1K-val` (its split is named `train`; the
contents are the validation set). Download it once with

    huggingface-cli download mrm8488/ImageNet1K-val --repo-type dataset \
        --local-dir $PTA_DATA/imagenet

Labels follow the sorted-synset order of `imagenet_classes.txt`.
"""
import glob
import io
import os

import torch
from torch.utils.data import Dataset

from .. import config as C
from .tasks import _cifar100_classes, _root

IMAGENET_CLASSES = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")


def imagenet_prompts():
    names = [ln.rstrip("\n").split("\t", 1)[1] for ln in open(IMAGENET_CLASSES)]
    return [f"a photo of a {n}." for n in names]


class CIFAR100Control(Dataset):
    """CIFAR-100 test images, `per_class` per class drawn with one seeded generator."""

    def __init__(self, transform, per_class=C.CONTROL_PER_CLASS, seed=C.SEED):
        from torchvision.datasets import CIFAR100
        self.ds = CIFAR100(root=_root("cifar100"), train=False, download=True,
                           transform=transform)
        y = torch.tensor(self.ds.targets)
        g = torch.Generator().manual_seed(seed)
        idx = []
        for c in range(100):
            cand = (y == c).nonzero().flatten()
            idx.append(cand[torch.randperm(len(cand), generator=g)[:per_class]])
        self.idx = torch.cat(idx).tolist()
        self.prompts = ["a photo of a {}.".format(c) for c in _cifar100_classes()]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.ds[self.idx[i]]


class ImageNetControl(Dataset):
    """ImageNet validation images, `per_class` per class drawn with one seeded generator.

    Rows are taken in the global order of the sorted parquet shards; for each class
    in label order a permutation from one shared generator picks the images."""

    def __init__(self, transform, per_class=C.CONTROL_PER_CLASS, seed=C.SEED, root=None):
        import pyarrow.parquet as pq
        root = root or os.path.join(C.DATA_DIR, "imagenet")
        self.files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"no ImageNet parquet shards under {root}; see pta/data/controls.py")
        self.transform = transform
        per = {}
        for fi, f in enumerate(self.files):
            labels = pq.ParquetFile(f).read(columns=["label"]).column("label").to_pylist()
            for ri, c in enumerate(labels):
                per.setdefault(c, []).append((fi, ri))
        g = torch.Generator().manual_seed(seed)
        self.picks = []
        for c in sorted(per):
            rows = sorted(per[c])
            sel = torch.randperm(len(rows), generator=g)[:per_class]
            self.picks += [rows[i] + (c,) for i in sel.tolist()]
        self._tables = {}
        self.prompts = imagenet_prompts()

    def _table(self, fi):
        if fi not in self._tables:
            import pyarrow.parquet as pq
            self._tables[fi] = pq.ParquetFile(self.files[fi]).read(columns=["image"])
        return self._tables[fi]

    def __len__(self):
        return len(self.picks)

    def __getitem__(self, i):
        from PIL import Image
        fi, ri, c = self.picks[i]
        b = self._table(fi).column("image")[ri]["bytes"].as_py()
        return self.transform(Image.open(io.BytesIO(b)).convert("RGB")), c


def control_sets(transform, names=C.CONTROLS):
    """{name: dataset} for the requested controls, each carrying `.prompts`."""
    out = {}
    for n in names:
        out[n] = CIFAR100Control(transform) if n == "cifar100" else ImageNetControl(transform)
    return out
